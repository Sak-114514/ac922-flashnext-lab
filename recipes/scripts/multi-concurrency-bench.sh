#!/usr/bin/env bash
# multi-concurrency-bench.sh — 多并发叠加实验执行器（设计稿，2026-10-04）
#
# 部署：scp 到 ac922:/home/debian/ac922-llama-next/scripts/ 后在 ac922 上以 debian 用户运行。
# 用法：
#   ./multi-concurrency-bench.sh --list                 # 列出场景
#   ./multi-concurrency-bench.sh --scenario A2          # 单场景
#   ./multi-concurrency-bench.sh --scenario A1,A2,A3    # 顺序多场景
#   ./multi-concurrency-bench.sh --scenario A3 --agent-turns 4   # A3 波次后追加 cache_prompt 多轮 agent 模式
#   REF_DIR=manifests/zcode-xxx ./multi-concurrency-bench.sh --scenario B1   # 指定 parity 参考目录
#
# 场景语义见 concurrency/PLAN.md。纪律：
#   - 每个场景一次全新 systemd unit + 全新 label + 全新日志（防追加污染）；
#   - matrix_runner 自带 58C 守护与 /slots 校验；parity 参考目录可选（--reference-directory）；
#   - 本脚本不编译、不改源码、不动生产服务；跑完即停测试 unit。
set -euo pipefail

ROOT=${AC922_ROOT:-/home/debian/ac922-llama-next}
SCRIPTS="$ROOT/scripts"
OUTROOT="$ROOT/concurrency"
UNIT=ac922-flashnext-test
TS=$(date +%Y%m%d-%H%M%S)

die() { echo "FATAL: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 预检：vLLM 复活 + GPU 残留显存（运维红线，见 GPT-HANDOVER.md §4）
# ---------------------------------------------------------------------------
preflight() {
  systemctl --user stop ac922-coding-api.service ac922-acestep.service 2>/dev/null || true
  sleep 2
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
  echo "preflight: total GPU memory.used = ${used} MiB"
  [[ "$used" -lt 600 ]] || die "GPU 显存未清零（${used} MiB）——先查 ac922-coding-api.service 是否复活"
  # 6 卡在线检查
  local n
  n=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
  [[ "$n" == "6" ]] || die "期望 6 卡，实际 ${n}"
}

stop_server() {
  local loaded
  loaded=$(systemctl --user show "$UNIT.service" -p LoadState --value 2>/dev/null || echo not-found)
  [[ "$loaded" == "not-found" ]] || systemctl --user stop "$UNIT.service" || true
}

# ---------------------------------------------------------------------------
# 公共 env（以 manifests/island-hb-multi + island-h-fifo-single 的并集为底，
# 去掉未验项，由各场景叠加）。所有值与已验 preset 一致，不要随手改。
# ---------------------------------------------------------------------------
base_env() {
  cat <<'EOF'
AC922_TEST_PLE_SHARD=0
AC922_TEST_MTP=1
AC922_TEST_MTP_DEPTH=4
AC922_TEST_MTP_PAD=6
AC922_TEST_MTP_UBATCH=32
AC922_TEST_CUDA_GRAPHS=1
AC922_TEST_BATCH=2048
AC922_TEST_HC_FUSED=1
AC922_TEST_HTTP_THREADS=8
AC922_TEST_NCCL_AUTO=1
AC922_TEST_NUMA0=1
LLAMA_MTP_FOLD=1
LLAMA_GPU_ARGMAX=1
GGML_CUDA_VOLTA_GDN_X4=1
GGML_CUDA_VOLTA_Q8_HEAD_TC=0
LLAMA_QSA_SPARSE_KERNEL=1
LLAMA_QSA_GATHER_MAX_TOKENS=32
LLAMA_QSA_PATH_TRACE=1
LLAMA_STEP_TIMING=1
LLAMA_STEP_TIMING_PERIOD=32
LLAMA_SPEC_TIMING=1
LLAMA_DRAFT_TIMING=1
GGML_FT_HOST_TIMING=1
NCCL_DEBUG=INFO
NCCL_DEBUG_SUBSYS=INIT,TUNING
GGML_CUDA_MOE_GROUPED=1
LLAMA_DRAFT_DETP=0
GGML_CUDA_FATTN_ONECAT=0
LLAMA_PLE_ONECAT=0
LLAMA_MONOMOE_SM70=0
AC922_TEST_P2P_AR=0
EOF
}

# FIFO/prefill 组合（island-h-fifo-single 已验单路；多路未验 → 只进带 parity 门场景）
fifo_env() {
  cat <<'EOF'
GGML_META_ASYNC_COPY=1
GGML_META_FIFO_PIPELINE=1
LLAMA_PREFILL_KV_PAD=8192
EOF
}

island_env() { printf 'LLAMA_ISLAND_SPLIT=3,3\nCUDA_DEVICE_ORDER=PCI_BUS_ID\nAC922_TEST_GPU_ORDER=0,1,2,3,4,5\n'; }

scenario_env() {
  case "$1" in
    # A1: 6-way TP + B+H，2×64K 复现锚点（bh-2x64k 77.9 tok/s 口径的当代复测）
    A1)
      base_env
      printf 'AC922_TEST_GPU_ORDER=0,4,2,1,3,5\nLLAMA_QSA_POOLED_MULTI=1\nLLAMA_QSA_MULTI_MIN_CTX=32768\nAC922_TEST_CTX=266240\nAC922_TEST_SLOTS=2\nAC922_TEST_SLOT_CTX=132096\nAC922_TEST_UBATCH=512\n'
      ;;
    # A2: 岛 3,3 + H + B(长度门默认) —— 长C+B 从未验证过，本场景即首验
    A2)
      base_env; island_env
      printf 'LLAMA_QSA_POOLED_MULTI=1\nLLAMA_QSA_MULTI_MIN_CTX=32768\nAC922_TEST_CTX=266240\nAC922_TEST_SLOTS=2\nAC922_TEST_SLOT_CTX=132096\nAC922_TEST_UBATCH=512\n'
      ;;
    # A3: A2 全开 + FIFO 预填充组合（多路未验，parity 门）
    A3)
      base_env; island_env; fifo_env
      printf 'LLAMA_QSA_POOLED_MULTI=1\nLLAMA_QSA_MULTI_MIN_CTX=32768\nAC922_TEST_CTX=266240\nAC922_TEST_SLOTS=2\nAC922_TEST_SLOT_CTX=132096\nAC922_TEST_UBATCH=512\n'
      ;;
    # B1: 岛 C+H，B 关闭（8K 已证 B 劣化，短上下文主线不带 B）
    B1)
      base_env; island_env
      printf 'LLAMA_QSA_POOLED_MULTI=0\nAC922_TEST_CTX=266240\nAC922_TEST_SLOTS=4\nAC922_TEST_SLOT_CTX=132096\nAC922_TEST_UBATCH=256\nLLAMA_PREFILL_FAIR_SHARE=1\n'
      ;;
    # B2: B1 + B 强制开（MIN_CTX=0）——8K 劣势两臂对比的 B 开臂
    B2)
      base_env; island_env
      printf 'LLAMA_QSA_POOLED_MULTI=1\nLLAMA_QSA_MULTI_MIN_CTX=0\nAC922_TEST_CTX=266240\nAC922_TEST_SLOTS=4\nAC922_TEST_SLOT_CTX=132096\nAC922_TEST_UBATCH=256\nLLAMA_PREFILL_FAIR_SHARE=1\n'
      ;;
    # B3: B1 + FIFO 组合（多路未验，parity 门）
    B3)
      base_env; island_env; fifo_env
      printf 'LLAMA_QSA_POOLED_MULTI=0\nAC922_TEST_CTX=266240\nAC922_TEST_SLOTS=4\nAC922_TEST_SLOT_CTX=132096\nAC922_TEST_UBATCH=256\nLLAMA_PREFILL_FAIR_SHARE=1\n'
      ;;
    # C2: 6-way + H + B（6×8K 聚合 49152 ≥ 32768 → 长度门自动放行 B）——先跑，作 C1 的 parity 参考
    C2)
      base_env
      printf 'AC922_TEST_GPU_ORDER=0,4,2,1,3,5\nLLAMA_QSA_POOLED_MULTI=1\nLLAMA_QSA_MULTI_MIN_CTX=32768\nAC922_TEST_CTX=98304\nAC922_TEST_SLOTS=6\nAC922_TEST_SLOT_CTX=16384\nAC922_TEST_UBATCH=256\nLLAMA_PREFILL_FAIR_SHARE=1\nGGML_CUDA_GRAPH_VARIANTS_CAP=8\nAC922_TEST_HTTP_THREADS=12\n'
      ;;
    # C1: 岛 + H + B（同上，门自动放行）——极限 6×8K 主场景
    C1)
      base_env; island_env
      printf 'LLAMA_QSA_POOLED_MULTI=1\nLLAMA_QSA_MULTI_MIN_CTX=32768\nAC922_TEST_CTX=98304\nAC922_TEST_SLOTS=6\nAC922_TEST_SLOT_CTX=16384\nAC922_TEST_UBATCH=256\nLLAMA_PREFILL_FAIR_SHARE=1\nGGML_CUDA_GRAPH_VARIANTS_CAP=8\nAC922_TEST_HTTP_THREADS=12\n'
      ;;
    *) die "未知场景 $1（--list 查看）" ;;
  esac
}

scenario_cases() {
  case "$1" in
    A1|A2|A3) echo "2:65536" ;;
    B1|B3)    echo "2:8192,4:8192" ;;
    B2)       echo "2:8192" ;;
    C1|C2)    echo "4:8192,6:8192" ;;
  esac
}

# 推荐的 parity 参考链（也可用 REF_DIR 覆盖）：A2/A3→A1 产物；B2/B3→B1；C1→C2
scenario_default_ref() {
  case "$1" in
    A2|A3) echo "${REF_A:-}" ;;
    B2|B3) echo "${REF_B:-}" ;;
    C1)    echo "${REF_C:-}" ;;
    *)     echo "" ;;
  esac
}

# run_scenario <sc> [keep]   keep=1 时波次后不停 server（供 agent-turns 接续）
run_scenario() {
  local sc=$1 keep=${2:-0} label runner_args=()
  local label_id="conc-${sc}-${TS}"
  local outdir="$OUTROOT/$label_id"
  echo "=== scenario $sc -> $outdir ==="
  stop_server
  mkdir -p "$outdir" "$ROOT/logs"
  rm -f "$ROOT/logs/${label_id}.log"   # 同标签重跑必须清旧 log（运维红线 #3）

  local envfile; envfile=$(mktemp)
  scenario_env "$sc" > "$envfile"

  local cmd
  cmd=(systemd-run --user "--unit=$UNIT" --collect
    --property=WorkingDirectory="$ROOT" --property=TimeoutStopSec=90
    --property=KillSignal=SIGINT --property=RuntimeMaxSec=14400
    --property=StandardOutput=append:"$ROOT/logs/${label_id}.log"
    --property=StandardError=append:"$ROOT/logs/${label_id}.log")
  while IFS='=' read -r k v; do cmd+=("--setenv=$k=$v"); done < "$envfile"
  cmd+=(/bin/bash "$SCRIPTS/run_test_server.sh")
  "${cmd[@]}"
  echo "server dispatched: $label_id (log: logs/${label_id}.log)"

  runner_args=(--label "$label_id" --cases "$(scenario_cases "$sc")" --directory "$outdir")
  local ref; ref=$(scenario_default_ref "$sc")
  [[ -z "$ref" ]] || runner_args+=(--reference-directory "$ROOT/$ref")
  echo "runner: matrix_runner.py ${runner_args[*]}"

  if python3 "$SCRIPTS/matrix_runner.py" "${runner_args[@]}"; then
    echo "scenario $sc: waves OK"
  else
    echo "scenario $sc: FAILED —— 日志 $ROOT/logs/${label_id}.log，结果保留在 $outdir" >&2
    stop_server
    return 1
  fi

  summarize "$outdir" "$label_id"
  echo "$label_id" >> "$OUTROOT/completed-labels.txt"
  rm -f "$envfile"
  [[ "$keep" == 1 ]] || stop_server
}

# 汇总：逐 case 打 per-flow PP/TG + 聚合（server 口径求和 + wave 墙钟），追加 TSV
summarize() {
  local dir=$1 label=$2
  python3 - "$dir" "$label" <<'PYEOF'
import json, sys
from pathlib import Path
d, label = Path(sys.argv[1]), sys.argv[2]
for f in sorted(d.glob('c*-p*.json')):
    r = json.loads(f.read_text())['runs'][-1]
    if r.get('status') != 'ok':
        print(f'{label} {f.name}: status={r.get("status")} (见错误细节)'); continue
    pps = [round(x['prompt_tps'], 2) for x in r['flows']]
    tgs = [round(x['decode_tps'], 2) for x in r['flows']]
    agg = r['aggregate']
    print(f'{label} {f.name}: PP={pps} TG={tgs} '
          f'agg_decode_sum={agg["decode_tps_sum_per_flow_server"]:.2f} '
          f'agg_wave_wall={agg["decode_tps_wave_wall"]:.2f} '
          f'wall={agg["wave_wallsec"]:.1f}s')
    with open(d.parent / 'RESULTS.tsv', 'a') as tsv:
        tsv.write(f'{label}\t{f.name}\t{r["concurrency"]}x{r["context_tokens_requested"]}\t'
                  f'{agg["decode_tps_sum_per_flow_server"]:.2f}\t{agg["decode_tps_wave_wall"]:.2f}\t'
                  f'PP={pps} TG={tgs}\n')
PYEOF
}

# ---------------------------------------------------------------------------
# agent-turns 模式：cache_prompt=true 多轮增量（模拟 Codex 双 agent 的第二轮以后）
# 依赖场景 A3（或手动 --base-url）已起的 server；每轮只追加 ~256 token 历史。
# ---------------------------------------------------------------------------
agent_turns() {
  local turns=${1:-4} base=${AGENT_BASE_URL:-http://127.0.0.1:8002}
  python3 - "$base" "$turns" "$SCRIPTS" <<'PYEOF'
import json, sys, time
sys.path.insert(0, sys.argv[3])
from benchmark_server import HttpJsonClient, build_flow_prompts
base, turns = sys.argv[1], int(sys.argv[2])
client = HttpJsonClient(base, 1800.0)
flows = build_flow_prompts(client, 65536, 2)
hist = {i: list(toks) for i, (toks, _) in enumerate(flows)}
slots = [s['id'] for s in client.get('/slots')][:2]
out = []
for turn in range(turns):
    for i in range(2):
        r = client.post('/completion', {
            'prompt': hist[i], 'n_predict': 256, 'stream': False,
            'cache_prompt': True, 'return_tokens': True,
            'temperature': 0.0, 'seed': 1234, 'ignore_eos': True,
            'id_slot': slots[i]})
        t = r['timings']
        out.append({'turn': turn, 'flow': i, 'prompt_tokens': len(hist[i]),
                    'cache_n': t.get('cache_n'), 'prompt_ms': t.get('prompt_ms'),
                    'delta_prefill_tok': len(hist[i]) - (t.get('cache_n') or 0),
                    'decode_tps': t.get('predicted_per_second')})
        hist[i] += r['tokens']          # agent 历史增长：下一轮 prompt = 本轮全量
print(json.dumps(out, indent=2))
with open('/home/debian/ac922-llama-next/concurrency/agent-turns.json', 'a') as f:
    f.write(json.dumps(out) + '\n')
PYEOF
}

main() {
  local scenarios=() agent=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --list)
        printf 'A1  6-way B+H 2×64K 复现锚点\nA2  岛+B+H 2×64K（长C+B首验）\nA3  A2+FIFO 全开\nB1  岛 C+H 2/4×8K（B关）\nB2  B1+B强制（8K两臂之B开臂）\nB3  B1+FIFO\nC2  6-way H+B 4/6×8K（C1参考）\nC1  岛 H+B 4/6×8K 极限并发\n'
        exit 0 ;;
      --scenario) IFS=',' read -ra scenarios <<< "$2"; shift 2 ;;
      --agent-turns) agent="$2"; shift 2 ;;
      *) die "未知参数 $1" ;;
    esac
  done
  [[ ${#scenarios[@]} -gt 0 ]] || die "需要 --scenario（--list 查看）"
  mkdir -p "$OUTROOT"
  preflight
  for sc in "${scenarios[@]}"; do
    # 最后一个场景若带 --agent-turns 则保活 server 供 agent 模式接续
    if [[ "$sc" == "${scenarios[-1]}" && "$agent" -gt 0 ]]; then
      run_scenario "$sc" 1
      echo "=== agent-turns (cache_prompt=true, ${agent} turns) on live ${sc} server ==="
      agent_turns "$agent"
      stop_server
    else
      run_scenario "$sc"
    fi
  done
  echo "ALL DONE. 汇总: $OUTROOT/RESULTS.tsv"
}

main "$@"
