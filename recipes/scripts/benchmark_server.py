#!/usr/bin/env python3
"""Decode-first benchmark harness for the local llama-server HTTP API.

The harness deliberately uses only the Python standard library.  It obtains a
real token sequence from /tokenize, sends that sequence to /completion, and
uses the server's timings object for prefill/decode rates.  Client wall time
is recorded separately because it includes HTTP and queueing overhead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SCHEMA_VERSION = 1
PROMPT_UNIT = (
    "This is a fixed, readable benchmark paragraph. It contains varied words, "
    "punctuation, and numbers so that the tokenizer sees ordinary model input. "
    "The server must process this paragraph as a fresh prompt for every flow.\n"
)


class BenchmarkError(RuntimeError):
    """An expected benchmark failure that should be retained in the JSON."""

    def __init__(self, message: str, *, details: Any = None) -> None:
        super().__init__(message)
        self.details = details


class ApiError(BenchmarkError):
    def __init__(self, method: str, path: str, status: int, body: Any) -> None:
        super().__init__(
            f"{method} {path} returned HTTP {status}",
            details={"method": method, "path": path, "status": status, "body": body},
        )
        self.method = method
        self.path = path
        self.status = status
        self.body = body


class HttpJsonClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        base_url = base_url.strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("--base-url must start with http:// or https://")
        self.base_url = base_url
        self.timeout = timeout

    def request(self, method: str, path: str, payload: Any = None) -> Any:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=body, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                status = response.status
        except HTTPError as exc:
            raw = exc.read()
            raise ApiError(method, path, exc.code, _decode_body(raw)) from exc
        except URLError as exc:
            raise BenchmarkError(
                f"{method} {path} failed: {exc.reason}",
                details={"method": method, "path": path, "reason": str(exc.reason)},
            ) from exc
        except TimeoutError as exc:
            raise BenchmarkError(
                f"{method} {path} timed out after {self.timeout:.1f}s",
                details={"method": method, "path": path, "timeout_sec": self.timeout},
            ) from exc
        if status < 200 or status >= 300:
            raise ApiError(method, path, status, _decode_body(raw))
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BenchmarkError(
                f"{method} {path} returned non-JSON data",
                details={"method": method, "path": path, "body": _decode_body(raw)},
            ) from exc

    def get(self, path: str) -> Any:
        return self.request("GET", path)

    def post(self, path: str, payload: Any) -> Any:
        return self.request("POST", path, payload)


def _decode_body(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _error_record(exc: BaseException) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": str(exc),
    }
    if isinstance(exc, BenchmarkError) and exc.details is not None:
        record["details"] = exc.details
    return record


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def _parse_gpu_memory(value: str | None) -> list[int] | None:
    if value is None:
        return None
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpu-memory-mib expects comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("--gpu-memory-mib expects positive integers")
    return values


def tokenize_exact(
    client: HttpJsonClient,
    target: int,
    flow_index: int = 0,
) -> tuple[list[int], dict[str, Any]]:
    """Build one readable flow prompt, then take exactly target server tokens."""
    prefix = "" if flow_index == 0 else f"Concurrent request identifier: stream {flow_index}.\n"
    repeats = 1
    content = prefix + PROMPT_UNIT
    token_response: Any = None
    while True:
        token_response = client.post(
            "/tokenize",
            {"content": content, "add_special": False, "parse_special": True},
        )
        tokens = token_response.get("tokens") if isinstance(token_response, dict) else None
        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
            raise BenchmarkError(
                "/tokenize response did not contain an integer tokens array",
                details={"response": token_response},
            )
        if len(tokens) >= target:
            break
        repeats *= 2
        content = prefix + PROMPT_UNIT * repeats

    selected = tokens[:target]
    if len(selected) != target:
        raise BenchmarkError(
            f"tokenizer produced {len(selected)} tokens, expected {target}",
            details={"target": target, "available": len(tokens)},
        )
    if len(set(selected)) < 2 or all(token == 0 for token in selected):
        raise BenchmarkError(
            "generated prompt is not varied ordinary text",
            details={"unique_token_count": len(set(selected))},
        )
    token_hash = hashlib.sha256(
        json.dumps(selected, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    source = {
        "flow_index": flow_index,
        "prefix": prefix,
        "unit": "fixed readable paragraph",
        "repeats_used": repeats,
        "tokenized_content_tokens": len(tokens),
        "selected_tokens": target,
        "token_sha256": token_hash,
        "token_sample_first": selected[:8],
        "token_sample_last": selected[-8:],
    }
    return selected, source


def build_flow_prompts(
    client: HttpJsonClient,
    target: int,
    concurrency: int,
) -> list[tuple[list[int], dict[str, Any]]]:
    """Tokenize one distinct prompt per flow while preserving flow zero exactly."""
    prompts = [tokenize_exact(client, target, flow_index) for flow_index in range(concurrency)]
    hashes = [source["token_sha256"] for _, source in prompts]
    if len(set(hashes)) != len(hashes):
        raise BenchmarkError(
            "concurrent flow prompts are not distinct",
            details={"flow_hashes": hashes},
        )
    return prompts


def _rate(tokens: int | None, milliseconds: float | None) -> float | None:
    if tokens is None or milliseconds is None or milliseconds <= 0:
        return None
    return float(tokens) / (float(milliseconds) / 1000.0)


def _as_number(value: Any) -> int | float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _prompt_count_classification(actual: int, requested: int, cache_n: int | None) -> str:
    if actual == requested:
        return "exact"
    if actual == requested + 1:
        return "bos_plus_one"
    if actual == requested - 1:
        return "bos_minus_one"
    if actual < requested and cache_n is not None and cache_n > 0:
        return "shrunk_by_cache"
    if actual < requested:
        return "shrunk_or_truncated"
    return "mismatch"


def _completion_flow(
    client: HttpJsonClient,
    prompt_tokens: list[int],
    prompt_source: dict[str, Any],
    flow_index: int,
    output_tokens: int,
    seed: int,
    slot_id: int,
    barrier: threading.Barrier,
) -> dict[str, Any]:
    flow: dict[str, Any] = {
        "flow_index": flow_index,
        "slot_id": slot_id,
        "requested_input_tokens": len(prompt_tokens),
        "requested_output_tokens": output_tokens,
        "prompt_source": prompt_source,
        "ok": False,
    }
    started: float | None = None
    try:
        barrier.wait(timeout=60.0)
        started = time.monotonic()
        response = client.post(
            "/completion",
            {
                "prompt": prompt_tokens,
                "n_predict": output_tokens,
                "stream": False,
                "cache_prompt": False,
                "return_tokens": True,
                "temperature": 0.0,
                "seed": seed,
                "ignore_eos": True,
                "id_slot": slot_id,
            },
        )
        ended = time.monotonic()
        flow["wallsec"] = ended - started
        flow["response_status"] = 200
        flow["server_response"] = {
            key: response[key]
            for key in ("id_slot", "stop_type", "truncated", "tokens_cached", "timings")
            if isinstance(response, dict) and key in response
        }
        timings = response.get("timings") if isinstance(response, dict) else None
        if not isinstance(timings, dict):
            raise BenchmarkError(
                "completion response did not contain timings",
                details={"response": response},
            )
        prompt_n = _as_number(timings.get("prompt_n"))
        predicted_n = _as_number(timings.get("predicted_n"))
        prompt_ms = _as_number(timings.get("prompt_ms"))
        predicted_ms = _as_number(timings.get("predicted_ms"))
        if prompt_n is None or predicted_n is None or prompt_ms is None or predicted_ms is None:
            raise BenchmarkError(
                "completion timings omitted prompt_n/prompt_ms/predicted_n/predicted_ms",
                details={"timings": timings},
            )
        prompt_n = int(prompt_n)
        predicted_n = int(predicted_n)
        prompt_ms = float(prompt_ms)
        predicted_ms = float(predicted_ms)

        content = response.get("content") if isinstance(response, dict) else None
        output_token_ids = response.get("tokens") if isinstance(response, dict) else None
        output_token_ids_valid = isinstance(output_token_ids, list) and all(
            isinstance(token, int) and not isinstance(token, bool) for token in output_token_ids
        )
        if output_token_ids_valid:
            output_token_ids_hash = hashlib.sha256(
                json.dumps(output_token_ids, separators=(",", ":")).encode("ascii")
            ).hexdigest()
        else:
            output_token_ids_hash = None
        content_valid = isinstance(content, str)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest() if content_valid else None
        cache_n_value = _as_number(timings.get("cache_n"))
        cache_n = int(cache_n_value) if cache_n_value is not None else None
        count_classification = _prompt_count_classification(prompt_n, len(prompt_tokens), cache_n)
        checks = {
            "truncated_false": response.get("truncated") is False,
            "response_id_slot_matches": response.get("id_slot") == slot_id,
            "prompt_n_exact": prompt_n == len(prompt_tokens),
            "predicted_n_exact": predicted_n == output_tokens,
            "return_tokens_integer_array": output_token_ids_valid,
            "return_tokens_count_matches_predicted_n": output_token_ids_valid and len(output_token_ids) == predicted_n,
            "content_string": content_valid,
        }
        failure_reasons: list[str] = []
        if not checks["truncated_false"]:
            failure_reasons.append("response_truncated" if response.get("truncated") is True else "missing_truncated_false")
        if not checks["response_id_slot_matches"]:
            failure_reasons.append("response_id_slot_mismatch")
        if not checks["prompt_n_exact"]:
            failure_reasons.append(f"prompt_n_{count_classification}")
        if not checks["predicted_n_exact"]:
            if response.get("stop_type") == "eos" or predicted_n < output_tokens:
                failure_reasons.append("early_eos_or_short_generation")
            else:
                failure_reasons.append("predicted_n_not_requested_output_tokens")
        if response.get("stop_type") == "eos":
            failure_reasons.append("early_eos_stop_type")
        if not checks["return_tokens_integer_array"]:
            failure_reasons.append("return_tokens_missing_or_invalid")
        elif not checks["return_tokens_count_matches_predicted_n"]:
            failure_reasons.append("return_tokens_count_mismatch")
        if not checks["content_string"]:
            failure_reasons.append("content_missing_or_invalid")
        flow.update(
            {
                "input_tokens": prompt_n,
                "output_tokens": predicted_n,
                "prompt_ms": prompt_ms,
                "predicted_ms": predicted_ms,
                "prompt_tps": _as_number(timings.get("prompt_per_second")) or _rate(prompt_n, prompt_ms),
                "decode_tps": _as_number(timings.get("predicted_per_second")) or _rate(max(0, predicted_n - 1), predicted_ms),
                "decode_rate_source": "server predicted_per_second, or (predicted_n-1)/predicted_ms fallback",
                "server_timings": timings,
                "prompt_count_check": {
                    "requested": len(prompt_tokens),
                    "actual_processed": prompt_n,
                    "cache_n": cache_n,
                    "delta": prompt_n - len(prompt_tokens),
                    "classification": count_classification,
                    "note": "BOS plus/minus one is classified explicitly; strict success still requires exact prompt_n",
                },
                "validation": {
                    "successful": not failure_reasons,
                    "checks": checks,
                    "failure_reasons": failure_reasons,
                },
                "output_token_ids": output_token_ids if output_token_ids_valid else None,
                "output_token_ids_sha256": output_token_ids_hash,
                "content_prefix_200": content[:200] if content_valid else None,
                "content_sha256": content_hash,
                "ok": not failure_reasons,
            }
        )
        if failure_reasons:
            flow["error"] = _error_record(
                BenchmarkError(
                    "completion response failed benchmark validation",
                    details={"failure_reasons": failure_reasons, "checks": checks},
                )
            )
    except BaseException as exc:  # retain every flow failure, including barrier failures
        if started is not None:
            flow["wallsec"] = time.monotonic() - started
        flow["error"] = _error_record(exc)
        if isinstance(exc, ApiError):
            flow["response_status"] = exc.status
    return flow


def _slot_ids(metadata: dict[str, Any], concurrency: int) -> tuple[list[int], bool]:
    raw_slots = metadata.get("slots")
    if isinstance(raw_slots, list):
        ids = [item.get("id") for item in raw_slots if isinstance(item, dict) and isinstance(item.get("id"), int)]
        if len(ids) < concurrency:
            raise BenchmarkError(
                f"server exposes {len(ids)} slots, but concurrency={concurrency} requires independent slots",
                details={"slot_ids": ids, "concurrency": concurrency},
            )
        return ids[:concurrency], True
    return list(range(concurrency)), False


def run_wave(
    client: HttpJsonClient,
    metadata: dict[str, Any],
    prompt_tokens: list[int],
    prompt_source: dict[str, Any],
    context_tokens: int,
    concurrency: int,
    output_tokens: int,
    seed: int,
    label: str,
    flow_prompts: list[tuple[list[int], dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if flow_prompts is None:
        flow_prompts = (
            [(prompt_tokens, prompt_source)]
            if concurrency == 1
            else build_flow_prompts(client, context_tokens, concurrency)
        )
    if len(flow_prompts) != concurrency:
        raise BenchmarkError(
            "flow prompt count does not match concurrency",
            details={"flow_prompts": len(flow_prompts), "concurrency": concurrency},
        )
    record: dict[str, Any] = {
        "label": label,
        "context_tokens_requested": context_tokens,
        "concurrency": concurrency,
        "output_tokens_requested": output_tokens,
        "seed": seed,
        "temperature": 0.0,
        "cache_prompt": False,
        "return_tokens": True,
        "ignore_eos": True,
        "input_tokens": len(prompt_tokens),
        "prompt_source": prompt_source,
        "flow_prompt_sources": [source for _, source in flow_prompts],
        "slot_assignment_verified": False,
        "flows": [],
    }
    try:
        slot_ids, verified = _slot_ids(metadata, concurrency)
        record["slot_ids"] = slot_ids
        record["slot_assignment_verified"] = verified
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = _error_record(exc)
        return record

    barrier = threading.Barrier(concurrency)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="llama-bench") as executor:
        futures = [
            executor.submit(
                _completion_flow,
                client,
                flow_prompt_tokens,
                flow_prompt_source,
                flow_index,
                output_tokens,
                seed,
                slot_id,
                barrier,
            )
            for flow_index, ((flow_prompt_tokens, flow_prompt_source), slot_id) in enumerate(zip(flow_prompts, slot_ids))
        ]
        flows = [future.result() for future in as_completed(futures)]
    ended = time.monotonic()
    flows.sort(key=lambda item: item.get("flow_index", -1))
    record["flows"] = flows

    successful = [flow for flow in flows if flow.get("ok")]
    total_output = sum(int(flow["output_tokens"]) for flow in successful)
    per_flow_decode_sum = sum(float(flow["decode_tps"]) for flow in successful if flow.get("decode_tps") is not None)
    wave_seconds = ended - started
    record["aggregate"] = {
        "successful_flows": len(successful),
        "failed_flows": len(flows) - len(successful),
        "wave_wallsec": wave_seconds,
        "output_tokens": total_output,
        "decode_tps_wave_wall": total_output / wave_seconds if wave_seconds > 0 else None,
        "decode_tps_sum_per_flow_server": per_flow_decode_sum,
        "decode_tps_note": "wave_wall includes HTTP and queueing; sum_per_flow_server is not aggregate wall throughput",
    }
    try:
        record["slots_after"] = client.get("/slots")
    except BaseException as exc:
        record["slots_after_error"] = _error_record(exc)
    record["status"] = "ok" if len(successful) == concurrency else "failed"
    return record


def collect_metadata(client: HttpJsonClient, gpu_memory_mib: list[int] | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "base_url": client.base_url,
        "endpoint_errors": {},
        "gpu_vram_mib": gpu_memory_mib,
        "gpu_vram_note": (
            "provided by caller"
            if gpu_memory_mib is not None
            else "llama-server API does not expose per-GPU VRAM; supply --gpu-memory-mib or record nvidia-smi separately"
        ),
    }
    for name, method, path in (
        ("health", "get", "/health"),
        ("props", "get", "/props"),
        ("slots", "get", "/slots"),
    ):
        try:
            value = getattr(client, method)(path)
            metadata[name] = value
        except BaseException as exc:
            metadata["endpoint_errors"][name] = _error_record(exc)

    props = metadata.get("props")
    if isinstance(props, dict):
        default_settings = props.get("default_generation_settings")
        if isinstance(default_settings, dict):
            metadata["reported_slot_context_tokens"] = default_settings.get("n_ctx")
        metadata["reported_total_slots"] = props.get("total_slots")
    slots = metadata.get("slots")
    if isinstance(slots, list):
        metadata["slot_context_tokens"] = [item.get("n_ctx") for item in slots if isinstance(item, dict)]
    metadata["kv_capacity_interpretation"] = (
        "n_ctx from /props and each slot n_ctx from /slots are llama.cpp server context/KV capacity reports; "
        "they are not a vLLM token-pool measurement."
    )
    return metadata


def _append_output(path: str, document: dict[str, Any]) -> None:
    if path == "-":
        json.dump(document, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
        return
    output_path = Path(path)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            previous = json.load(handle)
        if isinstance(previous, dict) and isinstance(previous.get("runs"), list):
            previous["runs"].extend(document["runs"])
            document = previous
        elif isinstance(previous, list):
            previous.extend(document["runs"])
            document = {"schema_version": SCHEMA_VERSION, "runs": previous}
        else:
            raise BenchmarkError("--output file must contain a benchmark document or runs array")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="llama-server base URL")
    parser.add_argument("--context-tokens", nargs="+", type=_positive_int, default=[8192])
    parser.add_argument("--concurrency", type=_positive_int, default=1)
    parser.add_argument("--label", default="manual", help="server/configuration label, e.g. mtp or no-mtp")
    parser.add_argument("--output", default="-", help="JSON output path; '-' writes stdout")
    parser.add_argument("--output-tokens", type=_positive_int, default=512)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=_nonnegative_float, default=1800.0, help="per HTTP request timeout, max 1800s")
    parser.add_argument("--gpu-memory-mib", help="optional comma-separated per-GPU VRAM values to retain in JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.timeout > 1800.0:
        parser.error("--timeout cannot exceed 1800 seconds")
    try:
        gpu_memory = _parse_gpu_memory(args.gpu_memory_mib)
        client = HttpJsonClient(args.base_url, args.timeout)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))

    metadata = collect_metadata(client, gpu_memory)
    records: list[dict[str, Any]] = []
    for context_tokens in args.context_tokens:
        try:
            flow_prompts = build_flow_prompts(client, context_tokens, args.concurrency)
            prompt_tokens, prompt_source = flow_prompts[0]
            records.append(
                run_wave(
                    client,
                    metadata,
                    prompt_tokens,
                    prompt_source,
                    context_tokens,
                    args.concurrency,
                    args.output_tokens,
                    args.seed,
                    args.label,
                    flow_prompts,
                )
            )
        except BaseException as exc:
            records.append(
                {
                    "label": args.label,
                    "context_tokens_requested": context_tokens,
                    "concurrency": args.concurrency,
                    "output_tokens_requested": args.output_tokens,
                    "status": "failed",
                    "flows": [],
                    "error": _error_record(exc),
                }
            )

    document = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "label": args.label,
            "context_tokens": args.context_tokens,
            "concurrency": args.concurrency,
            "output_tokens": args.output_tokens,
            "seed": args.seed,
            "temperature": 0.0,
            "cache_prompt": False,
            "return_tokens": True,
            "ignore_eos": True,
            "timeout_sec": args.timeout,
        },
        "metadata": metadata,
        "runs": records,
    }
    try:
        _append_output(args.output, document)
    except BaseException as exc:
        print(f"failed to write benchmark JSON: {exc}", file=sys.stderr)
        return 1

    failed = any(record.get("status") != "ok" for record in records)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
