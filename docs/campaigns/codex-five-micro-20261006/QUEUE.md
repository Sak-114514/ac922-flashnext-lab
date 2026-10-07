# Five-lane microtest queue

Leader owns GPU execution. Five native executor agents own isolated preparation lanes. Baseline: immutable bundle-v1, source copies made before lane edits; original source/model/runtime/production and hardware untouched.

| Lane | First proof | GPU queue |
|---|---|---|
|01-multi-mtp|CPU pad/state ownership and actual multi-request harness|2/4 short requests, stable multi-verify padding; GPU chain initially inactive by existing guards|
|02-overlap|CPU two-frame lifecycle; actual GGML graph UID risk|two GPU CUDA frame/epoch fixture, then same-UID pointer-swap GGML fixture|
|03-ple-cache|real IQ4_NL rows/hash/dequant, bounded page I/O/LRU|CPU first; no invented GPU speed result|
|04-k8v4|actual quant/rotation reference and resource matrix|reuse lane1 client, Q8 K/V then K8V4 short1/2/4 streams as memory allows|
|05-dwdp|all seven main expert formats and exact owner/reassembly|one-island small raw expert peer prefetch ping-pong, checksum/event lifetime|

All model tests use unique transient user units, loopback port, request timeout and finally cleanup. Operator tests use unique logs, process timeout and process-only resources.58C and512MiB GPU margin gates; no held model while waiting. Build CUDA for sm70 explicitly, use existing CUDA525 wrapper. Do not install dependencies or toggle watchdog/fan/PSU/driver/kernel/power.

Do not infer integrated speed/quality from a fixture. Report new facts as measured, source-verified or pending. Graph-cache/pipeline tests must validate captured pointer changes; repeated real requests must include exact input SHA and request history. K8V4 fields do not prove a particular fused attention dispatch.
