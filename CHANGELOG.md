# Changelog

## 1.1.1
- Fixed an orphaned llama-server process being left running (holding VRAM and the port) when startup failed or timed out before the health check passed.
- Fixed "Port already in use" errors when changing the model or server settings with `keep_server_alive` enabled — the old server on that port is now restarted automatically. Temp-mode runs likewise replace a leftover kept-alive server instead of erroring.
- Registered an exit hook so kept-alive servers are stopped when ComfyUI shuts down, instead of being orphaned.
- Fixed a log-file handle leak when temporary (non-keep-alive) servers were stopped after each run.
- The Refresh Models button no longer silently resets a saved `model_path`/`mmproj_path` selection when the saved file isn't found in the rescanned folder (e.g. workflows shared across machines); the saved value is kept and validated at run time.
- llama-server is now explicitly bound to 127.0.0.1 so the API is never exposed on the local network.
- RGBA (4-channel) image tensors are converted to RGB before JPEG encoding instead of erroring.
- `disable_thinking` now sends both `thinking` and `enable_thinking` template kwargs, covering Qwen3/GLM-style templates as well.
- Empty system prompts are omitted from the request instead of sending an empty system message.
- On Windows, quoted paths in `extra_server_args` no longer pass literal quote characters to llama-server.
- Clarified that the idle-timeout check runs on the next node execution (there is no background timer).
- Added a new "CC Llama Vision Advanced Options" node holding sampling, performance, lifecycle, and diagnostics settings (temperature, top_p, top_k, min_p, repeat/presence/frequency penalty, seed, disable_thinking, stop_sequences, n_gpu_layers, ctx_size, threads, threads_batch, extra_server_args, keep_server_alive, idle_timeout_s, force_restart, startup_timeout_s, request_timeout_s, debug, server_log_path).
- CC Llama Vision now exposes only model selection, prompts, media, and max_tokens as direct widgets, plus a new optional `advanced_options` input that accepts the Advanced Options node's output. Leaving it disconnected preserves the previous defaults.
- This is a breaking change for saved workflows/API graphs that reference the moved widgets directly by name; reconnect an Advanced Options node (or re-add the workflow) to restore custom values.

## 1.0.3
- Added package metadata and dependency declarations for easier installation and review.
- Clarified the node's security-safe behavior in the README.
- Improved parsing of extra server arguments for safer startup handling.
