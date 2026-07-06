# CC Llama Vision

## Install
Copy this whole folder into `ComfyUI/custom_nodes/` as-is, e.g.:

```
ComfyUI/custom_nodes/cc_llama_vision/
    __init__.py
    cc_llama_vision.py
    js/
        llama_vision.js
```

Restart ComfyUI. The `js/` folder is served automatically via `WEB_DIRECTORY`.

## What changed and why

### 1. Dynamic models folder
`models_dir` is a normal text widget, plus a new **"🔄 Refresh Models"** button
on the node. Type/paste a folder, click Refresh — the `model_path` and
`mmproj_path` dropdowns repopulate immediately (no ComfyUI page reload). This
is done via a small frontend extension (`js/llama_vision.js`) that calls a new
backend route (`/llama_vision/scan_models`) which rescans the folder for
`.gguf` files.

The last folder you scanned is remembered in `llama_vision_config.json`
(next to the node file) and becomes the default next time ComfyUI starts, so
you only need to set it once.

(Previously `models_dir` did nothing — the dropdown always scanned a fixed
default folder, and `set_models_dir` was never called from anywhere. Also,
`models_dir` was declared as a required input but missing from `run()`'s
signature, which would have crashed with a `TypeError` the first time the
node actually executed.)

### 2. llama-server path — PATH-first
`llama_server_path` now defaults to the bare command `llama-server` whenever
it resolves on PATH (e.g. after `winget install llama.cpp`), instead of
resolving to a fixed absolute path. That way, future `winget upgrade`s that
move the install location keep working without touching this field. It only
falls back to the old hardcoded `C:\llama.cpp\llama-server.exe` if nothing is
found on PATH.

### 3. `seed` control_after_generate → fixed
Changed `"fixed": True` (not a real ComfyUI option — did nothing) to
`"control_after_generate": "fixed"`, so the widget's after-generate mode
actually defaults to Fixed instead of Randomize.

### 4. VRAM handoff between the LLM and an image model
llama-server runs as a separate OS process, so ComfyUI's model manager has no
idea it's holding VRAM — it can only manage models it loaded itself. If you
caption with the LLM and then generate an image with a large diffusion model
in the same session, the second step can OOM because llama-server never got
unloaded.

New node: **CC Llama Server Unload (free VRAM)**. Wire it in between:

```
CC Llama Vision (caption) ──> CC Llama Server Unload ──> CLIPTextEncode.text ──> KSampler ──> ...
```

The unload node passes the caption text straight through, so it's a
transparent pass-through — but because your image-generation branch now
*depends* on its output, ComfyUI's execution order guarantees the llama
server is killed (freeing VRAM) before the image model loads. It also has an
option to call ComfyUI's own `unload_all_models()` / `soft_empty_cache()` at
the same time, in case you want to be extra sure nothing lingers.

If you don't do this chaining, `keep_server_alive=False` (the new default)
still kills the temporary server after every single caption call, which is
safe but means a full model reload cost on every caption. `keep_server_alive`
+ the idle timeout is best for batches of captions run back-to-back; use the
Unload node right before you switch to image generation either way.
