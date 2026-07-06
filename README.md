# CC Llama Vision

A ComfyUI custom node pack that drives a local [`llama-server`](https://github.com/ggml-org/llama.cpp) instance (llama.cpp) to caption images, batches of images, or video frames using any GGUF vision-language model — with automatic server lifecycle management and VRAM-safe handoff to the rest of your workflow, so you never have to run `llama-server` by hand.

The main node launches `llama-server` for you, waits for it to become healthy, sends an OpenAI-compatible chat completion request with your image(s) embedded as base64 data URLs, and returns the model's text response as a `STRING` output ready to feed into the rest of your graph.

## Nodes in this pack

### CC Llama Vision
The core captioning node — starts/reuses `llama-server`, sends the prompt + media, returns the caption.

### CC Llama Server Unload (free VRAM)
A pass-through node for mixed LLM + image-generation workflows. `llama-server` runs as a separate OS process, so ComfyUI's model manager has no idea it's holding VRAM. Wire this node between the caption node and whatever consumes its text (e.g. a `CLIPTextEncode.text` input) — because your image-generation branch now depends on its output, ComfyUI's execution order *guarantees* the llama server is killed (freeing VRAM) before your diffusion model loads, instead of relying on timers or guesswork. It can optionally also call ComfyUI's own `unload_all_models()` / `soft_empty_cache()` at the same time.

```
CC Llama Vision (caption) → CC Llama Server Unload → CLIPTextEncode.text → KSampler → ...
```

## Features

- **Automatic server management** — starts `llama-server` with your chosen model and mmproj file, waits for `/health`, and reuses the same process across multiple node runs instead of restarting every time.
- **Keep-alive or one-shot modes** — enable `keep_server_alive` to keep the model resident in memory between runs, or leave it off (the default) to spin up a temporary server that's killed as soon as the request finishes.
- **Idle auto-unload** — set `idle_timeout_s` to automatically kill an idle persistent server after N seconds, freeing VRAM without any manual intervention.
- **Explicit VRAM handoff node** — the new `CC Llama Server Unload` node guarantees the LLM is unloaded before an image model loads next in the same session, solving the classic "LLM + diffusion model both resident → OOM" problem for two-step workflows (generate a prompt, then generate the image).
- **Force restart** — force-kill and relaunch the server on demand (e.g. after changing model files on disk) without waiting for an idle timeout.
- **Live, dynamic model discovery** — type any folder into `models_dir` and click **🔄 Refresh Models** to rescan for `.gguf` files on the spot; the `model_path` and `mmproj_path` dropdowns repopulate immediately, no ComfyUI page reload required. The folder you last used is remembered automatically (`llama_vision_config.json`) and becomes the default the next time ComfyUI starts.
- **PATH-aware executable discovery** — `llama_server_path` defaults to the bare `llama-server` command whenever it resolves on `PATH` (e.g. installed via `winget install llama.cpp`), so future upgrades that move the binary keep working with no changes needed. Falls back to a legacy fixed install path only if nothing is found on `PATH`.
- **Multiple input modes** — accepts a single `image`, a batched `image_batch`, or `video_frames`, and automatically downsamples long frame sequences to `max_video_frames` using evenly spaced sampling.
- **Video frame labeling** — optionally prefixes each sampled frame with `[Video frame N of total]` so the model can reason about temporal order.
- **Full sampler control** — exposes `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`, `seed` (defaults to ComfyUI's `control_after_generate: fixed`), and custom `stop_sequences`.
- **Thinking-mode toggle** — `disable_thinking` suppresses chain-of-thought/reasoning output on models that support it, so `caption` contains only the final answer.
- **Robust empty-output handling** — if a model hits `max_tokens` while still "thinking" and returns no content, the node surfaces the raw `reasoning_content` and `finish_reason` instead of failing silently.
- **Debug logging** — optional `debug` flag prints the raw JSON response, and all server stdout/stderr is captured to a configurable `server_log_path` for troubleshooting startup failures.
- **Extra server args** — pass any additional `llama-server` CLI flags (e.g. `--flash-attn`, custom rope settings) via `extra_server_args` without modifying the node.
- **Configurable timeouts** — separate `startup_timeout_s` (server boot) and `request_timeout_s` (inference) so slow model loads and long generations don't get killed prematurely.

## Inputs at a glance

| Category | Inputs |
|---|---|
| Server | `llama_server_path`, `models_dir` (+ 🔄 Refresh Models), `model_path`, `mmproj_path`, `port`, `n_gpu_layers`, `ctx_size`, `threads`, `threads_batch`, `extra_server_args` |
| Lifecycle | `keep_server_alive`, `idle_timeout_s`, `force_restart`, `startup_timeout_s` |
| Prompting | `system_prompt`, `user_prompt`, `stop_sequences`, `disable_thinking` |
| Sampling | `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`, `seed`, `max_tokens` |
| Media | `image`, `image_batch`, `video_frames`, `max_video_frames`, `label_video_frames` |
| Diagnostics | `debug`, `server_log_path`, `request_timeout_s` |

**CC Llama Server Unload** inputs: `trigger` (STRING, forced input — connect your caption output here), `target` (`all_servers` / `specific_port`), `port`, `also_free_comfyui_vram`.

## Outputs

- **CC Llama Vision** → `caption` (`STRING`) — the model's generated text response.
- **CC Llama Server Unload** → `trigger` (`STRING`) — the same text passed straight through, used only to enforce execution order.

## Install

Copy this folder into `ComfyUI/custom_nodes/` so it looks like:

```
ComfyUI/custom_nodes/ComfyUI-cc_llama_vision/
    __init__.py
    cc_llama_vision.py
    js/
        llama_vision.js
```

Restart ComfyUI. The `js/` folder is served automatically via `WEB_DIRECTORY` and adds the Refresh Models button to the node.

## Requirements

- A working `llama-server` build from llama.cpp, either on `PATH` (recommended — e.g. via `winget install llama.cpp`) or at a path you specify in `llama_server_path`.
- A GGUF model + matching `mmproj` GGUF file for vision support.
