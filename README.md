# LightLX

```
  _    _       _     _   _    __  __
 | |  (_) __ _| |__ | |_| |  \ \/ /   LightLX
 | |  | |/ _` | '_ \| __| |   \  /    run models too big for memory
 | |__| | (_| | | | | |_| |___/  \    (and the ones that fit, fast)
 |_____|_|\__, |_| |_|\__|_____/_/\_\
          |___/
```

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Platform: Apple Silicon](https://img.shields.io/badge/Platform-Apple%20Silicon-black.svg)
![Built on MLX](https://img.shields.io/badge/Built%20on-MLX-orange.svg)

**Run models too big for your RAM on Apple Silicon — any Hugging Face model that
fits on disk but not in memory — by streaming weights layer-by-layer.** Point it
at a model directory and go.

Three modes, auto-selected by size then architecture:
- **Resident (fast)** — if the model **fits in RAM**, it's loaded fully onto the GPU
  and runs at full speed, exactly like **LM Studio / Ollama**. No streaming. (Force
  streaming instead from `/menu` → Force stream.)
- **Generic streaming** — any mlx-lm-supported model **too big for RAM**, now **dense
  *and* MoE**. Dense (Llama, Qwen, Mistral, Gemma, Phi) streams layer-by-layer; MoE
  (Mixtral, Qwen2/3-MoE, OLMoE, DeepSeek, …) streams only the routed **top-k experts
  per token** — the scalpel, generalized to every arch that shares mlx-lm's `SwitchGLU`.
  Reuses mlx-lm's exact forward, so output is **bit-identical to native** (verified
  max|Δ logits| = 0 on both a dense and an MoE model). → [`lightlx/generic.py`](lightlx/generic.py)
- **GLM-5.2 streaming (flagship)** — a bespoke MoE-native engine for the full unquantized
  753B GLM-5.2 (~1.5 TB), with a resident skeleton + MLA/DSA specifics. → [`lightlx/model.py`](lightlx/model.py)

Resident is fast. Streaming is slow by design — tokens/sec ≈ disk-GB/s ÷
active-bytes-per-token: a 100 GB dense model on a 1 GB/s drive is ~tens of sec/token.
Physics, not a bug (see [Performance](#performance-the-honest-truth)).

```bash
pip install -e .          # once; then `lightlx` works from any directory
lightlx                   # asks for any model's path, then chat
```

```
○ direct · full ›  What is the capital of France?

working… (full BF16, streamed — slow · direct)
⠹ tok 1/512 · layer 45/78 · 0.72 GB/s · ~40s left      ← live spinner + ETA while it streams
…tokens stream in as they land…
  N tok · M min · ~0.01 tok/s · GB read                 ← summary
```
*(Timings illustrate the physics — ~tens of seconds per token. GLM-5.2 running
end-to-end is **confirmed** — see [Status](#status).)*

---

## Why this exists

No Apple Silicon machine — not even a 512 GB Mac Studio — can hold full BF16
GLM-5.2 in memory. So for *full precision* on a Mac, streaming from disk is the
**only** option, and the same pipeline gets faster for free as storage/RAM
improve. LightLX is that pipeline, built MoE-native.

## How it works

A transformer runs every layer for every token, but GLM-5.2 is a sparse
**Mixture-of-Experts**: each of the 75 MoE layers has 256 experts and a router
picks just **8 per token**. So per token you only need ~6% of the model — and
*which* 6% is decided on the fly. LightLX exploits exactly that:

- **No repacking** — reads individual tensors on demand straight from the original
  BF16 safetensors shards via direct `os.pread` (**not** `mmap`, which thrashes swap
  on a model bigger than RAM — see [DESIGN.md](docs/DESIGN.md)). One expert read
  touches only that expert. → [`lightlx/weights.py`](lightlx/weights.py)
- **The scalpel** — per MoE layer, run the router, load *only the 8 active
  experts*, never the other 248. ~16–32× less I/O than loading whole layers.
- **One reusable attention module** — instantiating the full model would
  allocate 1.5 TB of zeros, so we keep a single `StreamAttention` and `.update()`
  each layer's weights through it. → [`lightlx/model.py`](lightlx/model.py)
- **Exact upstream math** — reuses mlx-lm 0.31's `glm_moe_dsa`/`deepseek_v32`
  (MLA, the `noaux_tc` sigmoid router, RoPE, RMSNorm, the
  `kv_b_proj → embed_q/unembed_out` MLA absorption). We replaced only weight
  *residency*, not the numerics.
- **Resident skeleton** — embeddings, final norm, lm_head, router gates, norms,
  the 3 dense layers, and all 75 shared experts are pinned in RAM (~11 GB) so
  they never re-stream. `mx.eval` + `mx.clear_cache()` per layer keep peak RAM
  bounded; `set_wired_limit` (gentle, ~0.6×RAM) stops the OS swapping the pinned
  set without starving the UI.

## Using it — no flags, ever

Just run `lightlx`. It's fully input-driven:

```
$ lightlx
Recent models
  1  GLM-5.2                /Volumes/CP Drive/GLM-5.2
  2  Qwen2.5-0.5B-Instruct  …/models/Qwen2.5-0.5B-Instruct
Pick a number — or drag in / paste a model folder.  (q to quit)
› 2

  Qwen2.5-0.5B-Instruct · resident · 512 tokens max
  message the model, or /menu for settings · /help · /exit

resident  Qwen2.5-0.5B-Instruct ›  hello
…
```

- **Startup remembers your models.** Pick a recent one by number, or drag/paste a
  new folder. First run just asks for a folder.
- **`/menu`** opens a settings panel — reasoning on/off, reply length, switch
  model, fast mode (GLM), quit — all by typing a number. No commands to memorize.
- **Slash shortcuts** if you prefer: `/think`, `/tokens N`, `/model`, `/fast`,
  `/help`, `/exit`. `Ctrl-C` stops a reply.
- **It remembers across sessions.** Recent models + your preferences (reasoning,
  reply length) are saved to `~/.lightlx/state.json` on exit and restored next time.

There are **no flags**. Everything is a numbered menu or a question.

---

## Ollama, LM Studio, and the agent

`lightlx` can talk to **any local OpenAI-compatible server** — Ollama, LM Studio,
vLLM, llama.cpp — and run a full coding agent: files, shell, MCP, GitHub docs,
and **parallel subagents** for long-horizon work.

```
$ lightlx

Resume
  1  refactor cli.py           ollama/qwen · 2h ago

Recent
  2  qwen2.5-coder:32b         ollama

Connect
  3  Local MLX folder          stream or resident
  4  Ollama                    running · 6 models
  5  LM Studio                 running · 2 models
  6  Custom OpenAI URL         vLLM, llama.cpp, …
```

Pick a number. Local MLX then asks **Chat** or **Agent**. Ollama / LM Studio
go straight to the agent (and ask which folder is the workspace). If a server
isn't running, you get options: try another address, type a model name, or go back.

On a local MLX chat session, `/agent` switches into the full agent.

### What the agent can do

| Tool | Role |
|---|---|
| `read_file` / `write_file` / `edit_file` | Read, create, and patch files |
| `list_dir` / `glob` / `grep` | Navigate the workspace |
| `bash` | Run shell commands in the workspace |
| `fetch_url` | Pull a web page (HTML stripped to text) |
| `docs` | Read Claude Code, Codex, Ollama, MCP, or any `owner/repo` from GitHub |
| `skill` | Load a SKILL.md (Claude / Codex / LightLX skills) |
| `memory` | Read/write auto-memory that survives sessions |
| `task` | Spawn a **subagent** (`explore` / `implement` / `general`) |
| MCP tools | Anything you attach in `~/.lightlx/mcp.json` |

Several `task` calls in one turn run **in parallel** (Ollama / LM Studio). Use
`explore` for read-only research, `implement` to land a change, `general` for
everything else. Subagents cannot nest.

```
  → task  map the auth flow
  → task  find flaky tests
    ▸ subagent explore · map the auth flow
      → grep  login
      ✓ grep  login
    ✓ subagent explore · map the auth flow
```

Docs aliases: `claude-code`, `codex`, `ollama`, `lmstudio`, `mcp`, `lightlx` —
or pass any `owner/repo`. Example: *“read the Codex AGENTS.md and match that
style here.”*

### MCP

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "…" }
    }
  }
}
```

Put that at `~/.lightlx/mcp.json`, or `.lightlx/mcp.json` / `mcp.json` in the
workspace. `/mcp` shows what's connected; settings → Reload MCP picks up edits.
Each MCP tool is registered as `mcp_<server>_<tool>`.

Context is **auto-detected** from Ollama (`/api/show`, loaded `num_ctx`) and LM
Studio (`loaded_context_length`). The prompt line shows `used/window`. When the
session nears ~72% of the window, LightLX **compacts** older turns into a
handoff summary (or `/compact` anytime).

Sessions auto-save to `~/.lightlx/sessions/`. Resume from the startup list or `/resume`.

`/handoff` (or `/model`) switches Ollama ↔ LM Studio ↔ local MLX **without
dropping the chat**. If the new window is smaller, it compacts first.

### Instructions, skills, and memory

Same shape as Claude Code and Codex:

| What | Where | When it loads |
|---|---|---|
| Project instructions | `LIGHTLX.md`, `AGENTS.md`, `CLAUDE.md`, `.lightlx/LIGHTLX.md` | Every session |
| Personal / local | `LIGHTLX.local.md`, `~/.lightlx/LIGHTLX.md` | Every session |
| Rules | `.lightlx/rules/*.md`, `.claude/rules/*.md` | Every session (`paths:` frontmatter = on demand later) |
| Skills | `.lightlx/skills/<name>/SKILL.md`, `~/.lightlx/skills/`, plus `.claude/skills` and `.codex/skills` | Catalog always; body via `skill` or `/name` |
| Auto memory | `~/.lightlx/memory/<project>/MEMORY.md` | Index every session; topic files on demand |

`@path` imports work inside instruction files (like Claude). Skills follow the Agent Skills `SKILL.md` layout, including `!`command`` injection.

Claude and Codex skills are used as-is: LightLX reads `~/.claude/skills`, `~/.codex/skills`, and this repo’s `.claude/` / `.codex/` folders. `/import` copies or symlinks them into `~/.lightlx/skills` (or `.lightlx/skills`). A skill with `context: fork` runs as a **subagent** (same as `task`); several can run in parallel.

`/init` writes a starter `LIGHTLX.md`. `/skills` lists skills. `/memory` shows what the agent has saved. Say “remember we use pnpm” and it writes the memory store.

Bundled: `/code-review`, `/summarize-changes`.

Agent slash commands: `/tools` `/skills` `/memory` `/init` `/mcp` `/docs` `/workspace` `/task` `/compact` `/resume` `/save` `/handoff` `/clear` `/menu`.

### Fast mode — 4-bit skeleton, no download

`/menu` → Fast mode (or `/fast`). Quantizes the skeleton (attention/shared/dense/head)
to 4-bit **on load, from your own BF16 weights** (not a download), and keeps it
resident; the **routed experts stay full BF16**. Measured **~1.2×** on the M4 Pro.
GLM-5.2 only. Force-streaming (even when a model fits in RAM) is also in `/menu`.

## Run any model

Point LightLX at any mlx-lm-supported model directory — pick **Local MLX folder**
and drag/paste the path, or choose a remembered one. It checks the size: if it
**fits in RAM** it loads fully resident (fast, like LM Studio/Ollama); if it's
**too big** it streams from disk.

(Downloaded models can live anywhere; the repo keeps a `models/` folder for
convenience.)

How it works: LightLX builds the model via mlx-lm (MLX is lazy, so instantiating a
model far bigger than RAM costs ~nothing — the zero-params are never materialized),
loads the small resident set (embeddings, final norm, lm_head), and replaces each
decoder layer with a wrapper that **pages that layer's weights in from disk, runs
mlx-lm's real forward, frees them.** No model code to write — the architecture's
math comes from mlx-lm. Verified **bit-identical** to native mlx-lm.

**Dense and MoE both work.** For MoE models, paging *whole* layers would mean loading
every expert — most of which the router never picks. Instead LightLX swaps in a
`StreamingSwitchGLU`: each
MoE layer pages its router/attention/norms normally but loads **only the routed
top-k experts per token** from disk. Because nearly every mlx-lm MoE arch shares one
`SwitchGLU` module — and HF stores experts as separate per-expert tensors — this one
replacement covers Mixtral, Qwen2/3-MoE, OLMoE, DeepSeek, and more, bit-identically
(verified Δ=0 on OLMoE). v0 supports **uniform-MoE** models (every layer MoE); mixed
dense/MoE stacks raise a clear error, and GLM-5.2 keeps its bespoke engine.
Requirement: the model fits on your **disk** (not RAM), as safetensors + `config.json`.

## Performance: the honest truth

Single-stream decode is bound by `seconds/token = active-bytes ÷ disk-speed`, and
on 24 GB you can keep only ~6% of the model resident, so the rest streams every
token. We tried four software optimizations and **measured** all of them on the
real model:

| Optimization | Result | Why |
|---|---|---|
| Expert LRU cache | **~0% hit** | GLM routing is too diverse; 24 GB caches almost nothing |
| Async prefetch | **0.99× (no help)** | prefill is ~73% disk-*idle* — not I/O-bound to begin with |
| Gather/scatter MoE | **0.76× (slower)** | overhead-bound; for real prompts ~all 256 experts activate anyway |
| Read/compute overlap | **0.86× (slower)** | decode compute is µs/layer — nothing to hide the per-layer read behind; the read *is* the critical path |
| Fast mode (4-bit skeleton) | **~1.2×** | the only win — skeleton stops streaming |

**Conclusion:** no software trick is a big win on 24 GB + a 1 GB/s drive. The cost
is `(per-expert overhead + bytes) × (experts touched)`, and for any non-trivial
prompt the coupon-collector effect means **nearly all 256 experts activate per
layer** — there's no small per-prompt subset to exploit. Real speed comes from
*outside* this box:

1. **Thunderbolt-5 NVMe drive** (~5×, full precision) — cleanest win.
2. **4-bit experts** (~4×, needs ~400 GB disk) — shrink the part that dominates.
3. **Two-tier** — a small resident model for instant replies, 5.2 on demand —
   the only path to instant "hi" on 24 GB.

> **Why MoE-native matters:** streaming *whole* layers would read all 256
> experts/layer (~19 GB/layer, ~1.4 TB/token on GLM-5.2). LightLX reads only the 8
> active experts (~45 GB/token) → **~30× less I/O per token** — the difference
> between "runs in minutes" and "doesn't finish."

## Status

- ✅ **Resident mode**: models that fit in RAM load fully on the GPU and run fast (like LM Studio/Ollama) — auto-selected; ~80 tok/s on Qwen2.5-0.5B.
- ✅ **Generic engine — dense**: streams any mlx-lm dense model (Llama/Qwen/Mistral/Gemma/Phi/…) too big for RAM — verified **bit-identical** to native mlx-lm on Qwen2.
- ✅ **Generic engine — MoE scalpel**: streams uniform-MoE models (Mixtral/Qwen2-3-MoE/OLMoE/DeepSeek/…) loading only the routed top-k experts/token — verified **bit-identical** (Δ=0) to native mlx-lm on OLMoE → coherent text.
- ✅ **Ollama / LM Studio / OpenAI-compat**: full local agent — files, shell, MCP, GitHub docs, parallel subagents.
- ✅ **Conversation memory**: multi-turn history with context auto-trim, `/clear`, partial-reply capture on Ctrl-C — works in resident and streaming.
- ✅ Runs full unquantized GLM-5.2 on a 24 GB Mac (streaming, MoE-native).
- ✅ Streaming verified vs real shards; forward verified (finite logits, prefill
  + decode MLA paths, MoE routing + experts + shared); gather/scatter verified
  numerically identical to the reference.
- ✅ Interactive REPL: animated spinner + `layer N/78 · GB/s · ETA`, token
  counter, `/think`, `/fast`, `/clear`, Ctrl-C.
- ⚠️ **v0 limits:** context ≤ 2048 (DSA sparse-attention indexer bypassed — exact
  below `index_topk=2048`); greedy decoding only.
- ✅ **Confirmed end-to-end (2026-06-21):** full unquantized GLM-5.2 (1.4 TB, 753B
  MoE) on a 24 GB M4 Pro generated the correct answer — `The capital of France is`
  → **`Paris`** — streaming active experts from an external SSD with **zero swap
  thrash** (the pread loader holds at 1.5 TB scale). First token takes several
  minutes: prefill streams ~all experts across the 75 MoE layers on a ~1 GB/s drive.

## Roadmap

- **Two-tier CLI** — small resident / Ollama model answers instantly; full GLM-5.2 on `/big`. The day-to-day usability win.
- **Long context** — implement the DSA lightning indexer + shared-indexer wiring
  for > 2048 tokens (MLA already keeps the KV cache tiny).
- **Faster tier** — TB5 NVMe support; optional local 4-bit-expert quantizer.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full design notes: the architecture,
the `pread` swap fix, the MoE scalpel, the measured optimization log, and the
performance physics.

## Requirements

```
mlx >= 0.30 · mlx-lm >= 0.31.3 · transformers >= 5.0   (see requirements.txt)
```
Apple Silicon, macOS. Any mlx-lm-supported model as safetensors with a
`config.json` (single-file or sharded with `model.safetensors.index.json`),
sitting on a disk large enough to hold it.

## License

MIT — see [LICENSE](LICENSE). Built on [MLX](https://github.com/ml-explore/mlx)
and [mlx-lm](https://github.com/ml-explore/mlx-lm). Models you run carry their own
licenses.
