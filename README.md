# CC Llama Vision

CC Llama Vision is a ComfyUI custom node pack for captioning images, image batches, and video frames with GGUF vision-language models through a local llama.cpp server.

It manages the server lifecycle for you, so you can focus on prompt design and workflow composition instead of manually launching and stopping llama-server.

## What it does

The core node, CC Llama Vision, will:

- start or reuse a local llama-server process
- validate the selected model and mmproj files
- send the prompt plus image or video content to the server
- return the generated caption as a STRING output

An optional node, CC Llama Vision Advanced Options, holds the sampling, performance, lifecycle, and diagnostics settings and plugs into the main node's `advanced_options` input — leave it disconnected to just use sensible defaults.

A third node, CC Llama Server Unload, can be inserted later in the workflow to explicitly free VRAM by stopping the server before downstream image-generation steps run.

## Nodes

### CC Llama Vision
Use this node when you want to generate captions from:

- a single image
- a batch of images
- sampled video frames

It handles model/mmproj selection, prompts, media inputs, and direct connection to the rest of your graph. Connect a CC Llama Vision Advanced Options node to its `advanced_options` input for finer control over sampling and server behavior.

### CC Llama Vision Advanced Options
Optional node that bundles the less-frequently-tweaked settings into a single output, so the main node stays uncluttered:

- sampling: temperature, top_p, top_k, min_p, repeat_penalty, presence_penalty, frequency_penalty, seed, disable_thinking, stop_sequences
- performance: n_gpu_layers, ctx_size, threads, threads_batch, extra_server_args
- lifecycle: keep_server_alive, idle_timeout_s, force_restart, startup_timeout_s, request_timeout_s
- diagnostics: debug, server_log_path

Connect its `advanced_options` output to the main node's `advanced_options` input. If left disconnected, the main node falls back to the same defaults this node ships with.

### CC Llama Server Unload
Use this node as a VRAM-safe handoff point in mixed LLM + image-generation workflows.

It is especially useful when you want to ensure the llama-server process is stopped before a diffusion model loads later in the same graph.

```text
CC Llama Vision Advanced Options ─┐
                                   ├─→ CC Llama Vision → CC Llama Server Unload → CLIPTextEncode.text → KSampler → ...
                (media inputs) ────┘
```

## Key features

- automatic llama-server startup and health checks
- reuse of a persistent server for faster repeated runs
- optional keep-alive mode with idle timeout support
- force-restart support when model files change
- live model discovery with a Refresh Models button
- support for image, image_batch, and video_frames inputs
- sampling controls such as temperature, top_p, top_k, min_p, and penalties
- configurable stop sequences and thinking-mode toggling
- debug logging and configurable server log paths
- safe, reviewable packaging for ComfyUI Manager and the Registry

## Installation

### Option 1: manual install
Copy this repository into your ComfyUI custom nodes folder:

```text
ComfyUI/custom_nodes/ComfyUI-cc_llama_vision/
    __init__.py
    cc_llama_vision.py
    js/
        llama_vision.js
```

Restart ComfyUI.

### Option 2: install via ComfyUI Manager
If the node is published in the registry, install it from ComfyUI Manager as usual.

## Requirements

- a working llama-server binary from llama.cpp
- a GGUF vision-language model
- a matching mmproj GGUF file for vision support

The recommended setup is to make llama-server available on PATH, for example via the standard llama.cpp install flow on your platform.

## Inputs at a glance

### CC Llama Vision
- Server: llama_server_path, models_dir, model_path, mmproj_path, port
- Prompting: system_prompt, user_prompt, max_tokens
- Media: image, image_batch, video_frames, max_video_frames, label_video_frames
- Advanced: advanced_options (optional input from CC Llama Vision Advanced Options)

### CC Llama Vision Advanced Options
- Sampling: temperature, top_p, top_k, min_p, repeat_penalty, presence_penalty, frequency_penalty, seed, disable_thinking, stop_sequences
- Performance: n_gpu_layers, ctx_size, threads, threads_batch, extra_server_args
- Lifecycle: keep_server_alive, idle_timeout_s, force_restart, startup_timeout_s, request_timeout_s
- Diagnostics: debug, server_log_path

### CC Llama Server Unload
- trigger
- target (all_servers or specific_port)
- port
- also_free_comfyui_vram

## Security and registry notes

This node is designed to be safe and reviewable for ComfyUI Manager and the Registry:

- it does not use eval, exec, or other dynamic code execution paths
- it does not install Python packages at runtime
- it only launches a local external process when explicitly configured by the user
- required Python dependencies are declared in pyproject.toml and requirements.txt

## Notes

- The Refresh Models button is powered by the packaged frontend assets in the js folder.
- The last-used models directory is persisted for convenience across ComfyUI restarts.
- For long video sequences, frames are automatically sampled to stay within the configured frame limit.
