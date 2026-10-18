# Fleet MPK providers: GPT-OSS-120B persistent megakernel (gfx950, DPX+NPS2)

`src/gpt_oss_gfx950_mpk` is the source closure of the fleet **persistent megakernel** (MPK) for GPT-OSS-120B on
MI355X, one provider DSO per build configuration:

| config | DSO | what it runs |
|---|---|---|
| `ns` | `libaitermk_provider_gpt_oss_gfx950_fleet_mpk_ns.so` | single-stream decode, no speculation |
| `eagle3_b6` | `..._fleet_mpk_eagle3_b6.so` | EAGLE3 (LMSYS head) in the kernel: 6-token verify, up to 5 chained drafts, confidence gate 0.3 |
| `dflash_b8` | `..._fleet_mpk_dflash_b8.so` | DFlash (z-lab drafter) in the kernel: 8-token block, 7 drafts |

These are the exact builds behind the GSM8K-128 megakernel results: every `configs/<name>/kernel.hip` is that
build's generated kernel, `compile_flags.txt` is its compile command, and the DSO's device code is identical to the
standalone build's (`--save-temps` assembly compared line by line).

## How it differs from the Fleet decode provider

The Fleet decode provider (`gpt_oss_gfx950`, not part of this tree) is a decode kernel that vLLM launches every step over vLLM's own page-16 KV cache; prefill is stock
vLLM. The MPK providers are whole-request programs: the provider owns **prefill, its KV cache and its weights**,
runs one free-running launch per request, and streams tokens to the host from pinned memory (`MPK_TOK_HOST`).
vLLM keeps the API server, scheduler, detokenizer, request lifecycle and metrics. The KV cache stays in the
provider because its layout is part of the kernel's speed: head-major, 4096-token pages, each partition's heads in
that partition's memory. Binding vLLM's paged KV cache instead is the next step and needs kernel changes.

| | Fleet provider | Fleet MPK providers |
|---|---|---|
| prefill | stock vLLM | the provider (one token per pass; the spec builds 6 / 8 per pass) |
| KV cache | vLLM's paged KV, aliased | the provider's own |
| weights | packed from vLLM's loaded weights | the provider's exported state (below) |
| launch | one per decode step (or `FLEET_MK_PERSIST` tokens) | one per request |
| GPU | one device (SPX) | both DPX partitions of one MI355X (`MPK_MULTI_DEV=2`) |
| speculation | no | EAGLE3 and DFlash inside the kernel, bit-exact to non-speculative greedy |

## Requirements

| Piece | Value |
|---|---|
| GPU | MI355X (gfx950) in **DPX+NPS2**, stock amdgpu driver; the provider spans both partitions of one GPU |
| vLLM | 0.31.0 or 0.27.1 (vLLM's `--worker-cls` seam; no fork needed) |
| Sampling | greedy, `temperature=0`, one request at a time (`--max-num-seqs 1`), no logprobs |
| Compiler | ROCm 7.0.0 HIP (`FLEET_MPK_ROCM_ROOT`: AMD clang roc-7.0.0 25314); another compiler builds different, not bit-exact, device code. In vLLM's Ubuntu 22.04 image use an Ubuntu 22.04 build (e.g. `/opt/rocm-7.0.0` of `rocm/dev-ubuntu-22.04:7.0`): the 24.04 build's hipcc needs glibc 2.38 |
| Composable Kernel | the Fleet pin (`FLEET_CK_DIR`): `ROCm/composable_kernel` at `d8ee107a47d8485dbcffc79eb08e4f7c39ea6335`, the fleet megakernel repository's `deps/composable_kernel` |
| Provider state | the export of the same build (below), about 100 GiB per config |
| GPU monitoring | off while serving: AMD SMI metric queries (`amd-smi metric`, monitoring agents) stall the kernel |

## 1. Build the providers

```bash
git clone https://github.com/ROCm/composable_kernel /path/to/ck && git -C /path/to/ck checkout d8ee107a47d8485dbcffc79eb08e4f7c39ea6335
cmake -S aitermk/megakernels/fleet/src/gpt_oss_gfx950_mpk -B build/aitermk-fleet-mpk -GNinja \
  -DFLEET_MPK_ROCM_ROOT=/path/to/rocm-7.0.0 \
  -DFLEET_CK_DIR=/path/to/ck
cmake --build build/aitermk-fleet-mpk
```

`FLEET_MPK_CONFIGS` selects configs (default `ns;eagle3_b6;dflash_b8`); `FLEET_MPK_SAVE_TEMPS=ON` keeps each DSO's
device assembly. `FLEET_MPK_QUIET_FWDPASS` (default ON) adds `-DMPK_QUIET_FWDPASS` to every DSO: the kernel prints no
per-pass `[FWD_PASS]` lines, only its `[FWD_PASS_TOTAL]` summary (measured in section 4). Each DSO is compiled by `FLEET_MPK_ROCM_ROOT/bin/hipcc` with that root's headers and device
libraries and the config's recorded flags (a CMake custom command, not CMake's HIP language, which would substitute
the build environment's ROCm headers). Each exports only the `fleet_mpk_*` C ABI (`provider.exports`):

```c
int  fleet_mpk_abi_version(void);           // 1
const char *fleet_mpk_build_id(void);       // must equal the state's build_id
int  fleet_mpk_num_ptrs(void);  const char *fleet_mpk_ptr_name(int i);
int  fleet_mpk_init(void *const *ptrs, int num_ptrs, void *const *meta, int num_meta, void *profiler,
                    int rank, int num_workers, int num_local_schedulers, int num_remote_schedulers,
                    int max_seq_length, int total_num_requests, long long eos_token_id);
void fleet_mpk_set_rope_tables(void *cos, void *sin);
void fleet_mpk_set_max_seq_length(int n);   void fleet_mpk_init_request(void);
void fleet_mpk_reset_decode_progress(void); void fleet_mpk_launch(void *stream);  // blocks for the request
void fleet_mpk_request_stop(void);          // ends the running launch at its next 16-iteration check
int  fleet_mpk_decode_progress(int slot);   int  fleet_mpk_decode_token(int pos);
void fleet_mpk_finalize(void);
```

## 2. Provider state

The kernel's weights and buffers are in its own layout (MXFP4 repacks, experts split between the two dies, per-die
copies, the placement arena). The fleet exporter writes them once, from the Hugging Face checkpoint, in the same run
that generates `configs/<name>`: with `MPK_PROVIDER_EXPORT=<dir>`, `mirage.mpk.provider_export` writes
`<dir>/src` (this directory's `configs/<name>` inputs) and `<dir>/state`: `manifest.json` plus one file per device
allocation the kernel's pointers land in (torch segments and the cross-partition HIP VMM buffers), byte for byte.
The adapter recreates each allocation of the same size on the same partitions, loads it, and passes the new
addresses to `fleet_mpk_init`. A state belongs to exactly one build (`build_id`); the adapter refuses a mismatch.

To produce a state and its `configs/<name>`, run the exporter in the fleet megakernel repository,
github.com/sangeeta0201/fleet-chiplet-megakernel, branch `gpt-oss-120b-mi355x`. There
`scripts/nps2/aitermk/exp_run.sh ns|e3|df8` made the published exports (`ARENA=32769` for the MXFP4-LM-head 32K
builds; `LM=bf16 ARENA=466` for the BF16-LM-head builds, written to `<cfg>_bf16`), and
`README_GPT_OSS_120B_MI355X.md` lists the flag files, the container and the node setup.

Before it loads anything the adapter checks that every allocation file is complete, that the visible devices are
the partitions of one gfx950 GPU with the CUs the build's launch fills (128 each, DPX+NPS2), and that each has room
for its share of the state; each failure names its fix.

## 3. Serve

```bash
export AITER_MK_FLEET_VLLM=1 VLLM_PLUGINS=fleet_mk
export AITERMK_FLEET_MPK_PROVIDER=/abs/build/.../libaitermk_provider_gpt_oss_gfx950_fleet_mpk_ns.so
export AITERMK_FLEET_MPK_STATE=/abs/path/to/state/ns
export HIP_VISIBLE_DEVICES=0,1          # the two DPX partitions of one GPU
vllm serve /path/to/gpt-oss-120b --served-model-name gpt-oss-120b \
  --load-format dummy --enforce-eager --max-model-len 32768 --max-num-seqs 1 \
  --max-num-batched-tokens 4096 --no-enable-prefix-caching --no-async-scheduling \
  --gpu-memory-utilization 0.9 --worker-cls fleet_vllm.native.worker.FleetMPKWorker
```

`AITERMK_FLEET_MPK_PROVIDER` switches `fleet_vllm` from the Fleet decode mixin to the native provider path
(`fleet_vllm.native`): a weightless, attention-free stand-in model, a worker whose `execute_model` returns the
tokens the launch has produced, and a scheduler hook that accounts for several tokens per step. The engine log must
show `native fleet MPK provider registered`, `preflight ok`, `provider state ... loaded`, and
`provider ... initialized`.

A request's prompt plus `max_tokens` must stay below the build's token arena (`init_args.max_seq_length`, 32,769), so
each build serves up to 32,768 tokens and `--max-model-len` may be at most 32768; the server refuses to start with a
larger value, with any other flag above set otherwise, or with a watermark (`--watermark-config`) or custom logits
processors configured. Requests the kernel cannot decode (temperature above 0, logprobs of any kind, `n` > 1,
penalties, logit bias, structured output, `min_tokens`, a `thinking_token_budget`, forced decode tokens) are rejected
with HTTP 400 and the server keeps serving. When vLLM finishes a request early (EOS, a stop string, an abort or a
client disconnect), the adapter calls `fleet_mpk_request_stop` and the launch ends at its next 16-iteration check, so
the next request starts at once rather than after the request's remaining `max_tokens`.

The arena is fixed when a build is exported: it sizes the compiled kernel (`-DMPK_MAX_SEQ_LENGTH`, its attention
split) and the state's token buffer. The KV cache is paged in 4,096-token pages; the speculative builds list the
request's pages once per verify slot, so EAGLE3 (6 slots) holds 54 pages and DFlash (8 slots) 72. A longer context
needs a new export (the fleet exporter's `ARENA`), not a serving flag.

## 4. Evaluate

`engine_adapters/vllm/scripts/gpt_oss_build_and_repro.sh` builds the providers, serves each config through
`vllm serve` and runs the 128 questions with `scripts/gsm128_serve_bench.py` (greedy, `ignore_eos`, 256 tokens;
decode tok/s = tokens after the first / time from first to last streamed chunk), comparing every served answer
token by token with the published run. Its default configurations, `ns`, `eagle3_b6` and
`dflash_b8`, are the released MXFP4-LM-head 32K builds; `CONFIGS="ns_bf16 e3_bf16 df8_bf16"` runs the checkpoint's
BF16 LM head, the precision vLLM runs, in builds sized for these questions (the longest prompt plus 264 tokens).
`engine_adapters/vllm/tools/fleet_mpk_check.py` drives a provider without vLLM.

`engine_adapters/vllm/scripts/fleet_mpk_serve_robustness.py --provider SO --state DIR --refs JSON --model DIR`
checks a deployment in about 20 minutes, starting its own servers with the command above: the rejected requests
and edge requests (capacity, stop strings, disconnects, concurrent requests, chat, a random burst), a soak whose
every answer must equal its reference, SIGTERM and SIGKILL while a request decodes followed by a new server, and
each misconfigured startup. It prints one PASS/FAIL line per check and exits 1 on any failure. Measured on the
GSM8K-only builds: every check passes for EAGLE3 (both nodes) and DFlash; the no-speculation soak shows the known
issue below on one node. On the 32K no-speculation build the live group passes (30 checks), including three
early stops of full-capacity requests (a stop string, EOS, a client disconnect), after each of which the next
request completed in 0.1 s or less. On vLLM 0.31.0 the 32K EAGLE3 build passes all 49 checks (live, a 64-request
soak, restarts, startups).

### GSM8K-128 at matched precision, one request, one MI355X (DPX+NPS2), vLLM 0.31.0

| Config | megakernel, BF16 LM head: this command (Oct 19) | the same builds, megakernel's own plugin (Oct 18, two serves) | vLLM 0.31.0, AMD MXFP4 x FP8 checkpoint | megakernel / vLLM |
|---|---|---|---|---|
| No speculation | 877 | 879 (878, 880) | 248 | 3.5x |
| EAGLE3 (block 6, 5 drafts) | 1400 | 1410 (1404, 1416) | 527 | 2.7x |
| DFlash (block 8, 7 drafts) | 1212 | 1215 (1220, 1211) | 657 | 1.85x |

Output tokens per second through `vllm serve`, pooled over the 128 questions; the ratio uses the mean of the two
Oct 18 serves. Both sides run MXFP4 weights with FP8 inputs in the attention projections and the MoE and keep the
router, the LM head and the KV cache in BF16 ([PRECISION_MATCHED_COMPARISON.md](PRECISION_MATCHED_COMPARISON.md));
vLLM's rates are its offline per-step decode times. Every answer of this command's run is token-identical to the
Oct 18 serves (`scripts/data/gsm128_ref_<config>_bf16.json`, the first 100 tokens). The BF16 LM head reads 1.19 GB
per token instead of 0.32 GB: in the same builds it costs 11.6% without speculation and 5% / 3% with it (MXFP4 head
994 / 1484 / 1258).

### MXFP4 LM head: GSM8K-128, one request, one MI355X (DPX+NPS2, stock driver), Oct 2026

| Config | vLLM 0.31.0 | vLLM 0.27.1 | published (served, vLLM 0.27.1) | libAiterMK / stock vLLM 0.31.0 |
|---|---|---|---|---|
| No speculation | 969 | 964 | 991 | 3.6x |
| EAGLE3 (block 6, 5 drafts) | 1454 | 1451 | 1473 | 2.2x |
| DFlash (block 8, 7 drafts) | 1223 | 1222 | 1247 | 1.6x |

Output tokens per second through `vllm serve`, pooled over the 128 questions, for these 32K builds (Oct 17). The two
vLLM versions ran alternated on one GPU (0.27.1, 0.31.0, 0.27.1, 0.31.0, 0.27.1; means, every run within 1.5% of its
version's mean). Per round (accepted drafts plus the target's own token), both versions: EAGLE3 3.52 tokens in
2.46 ms, DFlash 4.25 tokens in 3.52 ms. Stock vLLM 0.31.0 on the same 128 questions: 268 / 651 / 784 (EAGLE3:
NVIDIA's Eagle3-v3 with 5 drafts, its best; DFlash: 7 drafts); vLLM 0.27.1 ran 268 / 562 / 615. At matched
precision, vLLM 0.31.0 with AMD's MXFP4 x FP8 checkpoint runs 248 / 527 / 657, and the megakernel with the
checkpoint's BF16 LM head 879 / 1410 / 1215, against 994 / 1484 / 1258 with its MXFP4 head in the same session
(vLLM 0.31.0; the GSM8K-length builds of the published column):
[PRECISION_MATCHED_COMPARISON.md](PRECISION_MATCHED_COMPARISON.md). Every served answer is
token-identical to the published run (the no-speculation references saved 100 tokens, the speculative ones all
256) except one no-speculation answer under vLLM 0.31.0, which the kernel itself produced (1 of 640 no-speculation
requests in these runs; see the known issue); no server restarted. The GSM8K-only builds they replace (466-token
arena) measured 990 / 1460 / 1241, and 982 / 1493 / 1255 from a fresh clone on a second node. Alternated on one GPU,
the 32K no-speculation build ran 964 tok/s against 988 and 989 for the GSM8K-only build (2.5% slower), and in a
second session on another GPU 973 against 983 and 983, where the 32K build without the stop flag ran 980; whether
the arena (attention compiled for up to 31 KV chunks instead of 8) or the stop flag costs that time is not resolved.
The speculative 32K builds run at the GSM8K-only builds' speed (same GPU, alternated: EAGLE3 2.461 against 2.478 and
2.485 ms per round, DFlash 3.528 against 3.531 and 3.535). Server processes of one build differ by about 1-2%.
On Oct 18, one serve per config of these builds on vLLM 0.31.0 (`gpt_oss_build_and_repro.sh`) read 966 / 1468 /
1236 (EAGLE3 served again: 1472), every question token-identical; on Oct 19, 963 / 1498 / 1243, and
961 / 1454 / 1224 from a fresh clone of this tree. Every GPT-OSS number of the blog, measured again on the latest code with
vLLM 0.31.0 on both sides, and how to reproduce each: [BLOG_REPRODUCTION.md](BLOG_REPRODUCTION.md).

Long context: the longest question's prompt (202 tokens) decoded for 32,566 tokens through `vllm serve` is
token-identical to the build's own standalone decode, all 32,566 tokens, for every config, on vLLM 0.27.1 and 0.31.0.
Speed over that decode (context 202 to 32,768): no speculation 817 tok/s, EAGLE3 720, DFlash 189 (vLLM 0.31.0 on
another GPU: 816, 722, 188). The speculative builds split attention over 5 (EAGLE3) and 3 (DFlash) KV chunks per
verify slot, against 31 without speculation, so at long context the no-speculation build is the fastest; on this
repetitive continuation DFlash also accepts few drafts.

Since Oct 17 the builds print no per-pass `[FWD_PASS]` lines (`FLEET_MPK_QUIET_FWDPASS`). Device printf goes through
the host: a launch printed one line for each of its first 10 passes and every 50th, and its whole per-pass log before
it could end, and the next request waited for that. Alternated on one GPU (vLLM 0.31.0, the 32K builds with and
without the flag, A/B/A/B/A, 128 questions per run), time to first token went from 168.2 to 137.0 ms without
speculation, from 68.1 to 62.6 ms for EAGLE3 and from 61.6 to 57.2 ms for DFlash, every run with the flag below every
run without it; decode speed did not change (979 against 980.5 tok/s, 1496 against 1497.5, 1257 against 1257).
Every answer was token-identical to the reference except one no-speculation answer in a run with the flag
(question 45, at token 18), which the kernel itself produced (the known issue below).

### GSM8K accuracy, all 1,319 test questions, Oct 17 2026

The three 32K providers (this tree's default build) next to stock vLLM's default engine with the real
checkpoint, in `vllm/vllm-openai-rocm:v0.31.0` (vLLM 0.31.0+rocm723, ROCm 7.2.3): greedy, one request at a time, a
freshly started server for every run, no failed request and a healthy endpoint after each run. Two formats, the same
questions in the same order:

| Config | vLLM's GSM8K evaluator: 5-shot completions, 256 tokens | Chat: 0-shot, reasoning low, up to 4,096 tokens |
|---|---|---|
| Stock vLLM | 56.1% | 95.4% |
| No speculation | 42.8% (-13.3) | 95.3% (-0.1) |
| EAGLE3 (block 6, 5 drafts) | 39.3% (-16.8) | 94.8% (-0.6) |
| DFlash (block 8, 7 drafts) | 35.5% (-20.6) | 95.3% (-0.1) |

Differences are in points, paired question by question against stock (95% interval about +-3.6 points on the
evaluator, +-1.0 in chat). In the chat format no difference is measurable (McNemar p = 1.0 / 0.29 / 1.0), and each
config's final answer equals stock's on 1,271-1,273 of the 1,319 questions. On the evaluator's raw few-shot
completions, which are not gpt-oss's format (vLLM's CI accepts 30% there for gpt-oss-20b), every gap is significant
(p < 0.001) and the completions differ from stock's from the first token or two.

That gap is the providers' precision. The checkpoint keeps the attention projections, router, embeddings and LM head in
BF16 (`modules_to_not_convert`), and stock vLLM runs them in BF16, with a W4A16 MXFP4 MoE. Fleet quantizes q/k/v/o and
the LM head to MXFP4 (about 12% relative weight error) and feeds every GEMM FP8 E4M3 inputs. The same quantization
applied inside stock vLLM (`engine_adapters/vllm/scripts/eval/fleetq_plugin`: fleet's own quantizers, bit-identical in
its `test_fq.py`; run eager with 64 requests at a time) takes the evaluator's score from 57.0% to 38.7% (MXFP4
weights) and 36.8% (plus FP8 inputs), into the providers' range; the questions both stock and the no-speculation
provider miss rise from 348 with the emulation off to 555-574 with it on. Keeping those layers in BF16 or FP8 is the
lever if the few-shot
completion score matters. The chat runs of stock and the emulation runs ran on a second MI355X in SPX/NPS1 mode;
everything else ran on the providers' GPU.

The score at matched precision (vLLM with AMD's MXFP4 x FP8 checkpoint), which part of the precision costs it
(the attention projections; FP6 or FP8 weights there recover it), the BF16 LM head's scores, MMLU-Pro in chat format
(no measurable loss against stock vLLM), why the speculative builds score lower on the evaluator (their attention is
split over fewer KV chunks; speculation itself is exact) and the software versions:
[PRECISION_MATCHED_COMPARISON.md](PRECISION_MATCHED_COMPARISON.md).

The evaluator cannot run all 1,319 questions at `--max-concurrency 1` as shipped: each request gets one 600 s aiohttp
total timeout, and the time a request waits for the single connection counts against it, so every request still
queued 600 s into the run fails with `TimeoutError` and is scored as an invalid answer. The 128-question gate fits in
600 s. The tools in `engine_adapters/vllm/scripts/eval/`:
- `geval_eval.py` runs the evaluator's own `main()` (prompts, stop strings and scoring unchanged) with that timeout
  raised, and records every question's completion;
- `gchat_eval.py` is the chat format above;
- `geval_run.py` starts a fresh server (stock or a provider), runs one of them, and rejects a run with any failed
  request or an unhealthy endpoint afterwards;
- `geval_sum.py` prints the table and the paired comparison (`ARMS`: the first is the baseline).

`--dev` takes HIP device ids (a provider uses both DPX partitions of its GPU), `--eval chat` selects the chat format,
and the run directory's name must end in `_gptoss_<arm>` for `geval_sum.py`:

```bash
# from the repository root
S=aitermk/engine_adapters/vllm/scripts/eval; R=results/vllm-evaluation; ID=$(date -u +%Y%m%dT%H%M%SZ)
EV=<vLLM fork>/tests/evals/gsm8k/gsm8k_eval.py
python $S/geval_run.py --mode stock --model <checkpoint> --dev 0 --out $R/${ID}_gptoss_stock \
  --evaluator $EV --datasets $R/datasets/gsm8k
python $S/geval_run.py --mode provider --provider <provider .so> --state <state dir> --model <checkpoint> \
  --dev 0,1 --out $R/${ID}_gptoss_ns --evaluator $EV --datasets $R/datasets/gsm8k
# stock vLLM with the providers' precision: FQ_W=1 MXFP4 weights, FQ_A=1 also FP8 inputs
PYTHONPATH=$S/fleetq_plugin FQ_W=1 FQ_A=1 python $S/geval_run.py --mode stock --plugins fleetq \
  --stock-args=--enforce-eager --conc 64 --model <checkpoint> --dev 0 --out $R/${ID}_gptoss_fqwa \
  --evaluator $EV --datasets $R/datasets/gsm8k
ARMS="stock ns fqwa" python $S/geval_sum.py $R
```

Identities: the evaluator from `proaditya/vllm_aitermk` `dd04b02` (`gsm8k_eval.py` SHA-256 `03409984a570fec3...`),
its GSM8K test file SHA-256 `3730d312f6e34405...`; provider SHA-256 `b5e43ec307e096da...` (ns), `c9172d3cca119e90...`
(eagle3_b6), `4742c64076dd8a01...` (dflash_b8).

## 5. Operational notes

- The builds launch the workers before the schedulers and retry a launch whose blocks are not all resident within
  1 s, so a launch cannot deadlock on placement.
- If a launch produces no token for `FLEET_MPK_STALL_S` seconds (default 120; a warning after 10), the engine stops
  with an error rather than hanging, so a supervisor can restart the server. AMD SMI metric queries stall the
  kernel for up to about 90 s; keep GPU monitoring off while serving.
- Rarely a launch stops making progress: every worker stays inside one iteration of the prompt, and the benchmark
  restarts the server after 30 s without a token. Both the 32K and the GSM8K-only builds do it: alternated on one GPU
  (Oct 16), the GSM8K-only builds had 5 server restarts in about 800 served launches and the 32K builds 2 in about
  530; on another GPU the next day, vLLM 0.31.0 and 0.27.1 alternated, none in about 2,000. One of 7 standalone
  export launches did the same, and a session loop without vLLM ran about 930 launches per no-speculation build
  without one. The cause is not identified yet; the stall watchdog turns such a launch into a restart.
- Loading a state takes 40-110 s from a network filesystem; restarts reuse the page cache.
- Stop a server by its process group; the engine core holds both partitions. SIGTERM or SIGKILL while a request
  decodes stops every server process within a second, and a new server can start at once.
- Run one server per node, with no other GPU work on the node. Measured on MI355X (DPX+NPS2, stock driver):
  while another process used a different GPU of the same node (a second server, or a plain PyTorch matmul loop),
  launches stalled for 15-140 s, and some reached the stall watchdog; starting or stopping that process can
  trigger a stall too. A server alone on its node ran thousands of launches without one.

### Known issue: rare divergence of the no-speculation build

On one GPU of one of the two test nodes, 14 of 4,049 no-speculation requests (0.35%, greedy, 100 tokens) differed from
the reference at a near-tie, mostly at the first content word of the answer; the kernel's own token record matched
what was streamed, so the adapter is not involved. Later runs found it on two more GPUs, one on each node: 2 of 512
requests (both in one run, right after a driver reload) and 1 of 640 (under vLLM 0.31.0, at token 64 of the answer),
and on that GPU the next day 1 of 640 again (question 45, at token 18).
The speculative builds still showed none (0 of about 4,900 checked). The cause is not yet identified; it predates the
launch fixes. The original fleet path (its own `demo.py` session, without libAiterMK) shows the same rate on the first
GPU (4 of 1,192 requests), so neither the adapter nor the provider export introduces it.


