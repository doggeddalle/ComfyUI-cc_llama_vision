# CC Llama Vision

A ComfyUI custom node pack that drives a local [`llama-server`](https://github.com/ggml-org/llama.cpp) instance from llama.cpp to caption images, image batches, or video frames using GGUF vision-language models. It includes automatic server lifecycle management and VRAM-safe handoff to the rest of a ComfyUI workflow, so users do not need to launch `llama-server` manually.

The main node launches `llama-server`, waits for it to become healthy, sends an OpenAI-compatible chat completion request with image input embedded as base64 data URLs, and returns the model's text response as a `STRING` output that can be fed directly into the rest of the graph.

## Nodes in this pack

### CC Llama Vision
The core captioning node. It starts or reuses `llama-server`, sends the prompt plus media, and returns the generated caption.

### CC Llama Server Unload (free VRAM)
A pass-through node for mixed LLM + image-generation workflows. `llama-server` runs as a separate OS process, so ComfyUI's model manager does not track its VRAM usage. Place this node between the caption node and whatever consumes its text output (for example, `CLIPTextEncode.text`). Because the downstream image-generation branch depends on this node's output, ComfyUI's execution order guarantees that the llama server is terminated and VRAM is freed before a diffusion model loads, rather than relying on timing-based cleanup.

```text
CC Llama Vision (caption) → CC Llama Server Unload → CLIPTextEncode.text → KSampler → ...
```

## Features

- **Automatic server management** — starts `llama-server` with the selected model and mmproj file, waits for `/health`, and reuses the same process across multiple node runs instead of restarting every time.
- **Keep-alive or one-shot modes** — enable `keep_server_alive` to keep the model resident in memory between runs, or leave it disabled to start a temporary server that is terminated as soon as the request finishes.
- **Idle auto-unload** — `idle_timeout_s` can automatically terminate an idle persistent server after a configured number of seconds, freeing VRAM without manual intervention.
- **Explicit VRAM handoff node** — the `CC Llama Server Unload` node guarantees that the LLM is unloaded before an image model loads later in the same session, which avoids the common out-of-memory case where both the LLM and diffusion model remain resident at the same time.
- **Force restart** — force-kills and relaunches the server on demand, for example after model files change on disk.
- **Live, dynamic model discovery** — enter any folder in `models_dir` and click **🔄 Refresh Models** to rescan for `.gguf` files immediately. The `model_path` and `mmproj_path` dropdowns are repopulated without a ComfyUI page reload. The last-used folder is stored in `llama_vision_config.json` and reused as the default on the next ComfyUI start.
- **PATH-aware executable discovery** — `llama_server_path` defaults to the bare `llama-server` command when it resolves on `PATH` (for example after `winget install llama.cpp`). This makes upgrades more resilient if the binary location changes. It falls back to a legacy fixed install path only when nothing is found on `PATH`.
- **Multiple input modes** — accepts a single `image`, a batched `image_batch`, or `video_frames`, and automatically downsamples long frame sequences to `max_video_frames` using evenly spaced sampling.
- **Video frame labeling** — can prefix sampled frames with `[Video frame N of total]` so the model can reason about temporal order.
- **Full sampler control** — exposes `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`, `seed`, and custom `stop_sequences`.
- **Thinking-mode toggle** — `disable_thinking` suppresses reasoning output on models that support it, so `caption` contains only the final answer.
- **Robust empty-output handling** — if a model reaches `max_tokens` while still generating reasoning and returns no final content, the node surfaces the raw `reasoning_content` and `finish_reason` instead of failing silently.
- **Debug logging** — an optional `debug` flag prints the raw JSON response, and server stdout/stderr can be captured to a configurable `server_log_path` for troubleshooting startup failures.
- **Extra server args** — additional `llama-server` CLI flags can be passed through `extra_server_args` without modifying the node.
- **Configurable timeouts** — separate `startup_timeout_s` and `request_timeout_s` values allow slow model loads and long generations to be handled independently.

## Inputs at a glance

| Category | Inputs |
|---|---|
| Server | `llama_server_path`, `models_dir` (+ 🔄 Refresh Models), `model_path`, `mmproj_path`, `port`, `n_gpu_layers`, `ctx_size`, `threads`, `threads_batch`, `extra_server_args` |
| Lifecycle | `keep_server_alive`, `idle_timeout_s`, `force_restart`, `startup_timeout_s` |
| Prompting | `system_prompt`, `user_prompt`, `stop_sequences`, `disable_thinking` |
| Sampling | `temperature`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `presence_penalty`, `frequency_penalty`, `seed`, `max_tokens` |
| Media | `image`, `image_batch`, `video_frames`, `max_video_frames`, `label_video_frames` |
| Diagnostics | `debug`, `server_log_path`, `request_timeout_s` |

**CC Llama Server Unload** inputs: `trigger` (STRING, forced input — connect the caption output here), `target` (`all_servers` / `specific_port`), `port`, `also_free_comfyui_vram`.

## Outputs

- **CC Llama Vision** → `caption` (`STRING`) — the model's generated text response.
- **CC Llama Server Unload** → `trigger` (`STRING`) — the same text passed straight through, used only to enforce execution order.

## Install

Copy this folder into `ComfyUI/custom_nodes/` so it looks like this:

```text
ComfyUI/custom_nodes/ComfyUI-cc_llama_vision/
    __init__.py
    cc_llama_vision.py
    js/
        llama_vision.js
```

Restart ComfyUI. The `js/` folder is served automatically via `WEB_DIRECTORY` and adds the Refresh Models button to the node UI.

## Requirements

- A working `llama-server` build from llama.cpp, either available on `PATH` (recommended, for example via `winget install llama.cpp`) or at a path specified in `llama_server_path`.
- A GGUF model and matching `mmproj` GGUF file with vision support.
