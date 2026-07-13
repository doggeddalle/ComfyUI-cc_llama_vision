# Changelog

## 1.1.0
- Added a new "CC Llama Vision Advanced Options" node holding sampling, performance, lifecycle, and diagnostics settings (temperature, top_p, top_k, min_p, repeat/presence/frequency penalty, seed, disable_thinking, stop_sequences, n_gpu_layers, ctx_size, threads, threads_batch, extra_server_args, keep_server_alive, idle_timeout_s, force_restart, startup_timeout_s, request_timeout_s, debug, server_log_path).
- CC Llama Vision now exposes only model selection, prompts, media, and max_tokens as direct widgets, plus a new optional `advanced_options` input that accepts the Advanced Options node's output. Leaving it disconnected preserves the previous defaults.
- This is a breaking change for saved workflows/API graphs that reference the moved widgets directly by name; reconnect an Advanced Options node (or re-add the workflow) to restore custom values.

## 1.0.3
- Added package metadata and dependency declarations for easier installation and review.
- Clarified the node's security-safe behavior in the README.
- Improved parsing of extra server arguments for safer startup handling.
