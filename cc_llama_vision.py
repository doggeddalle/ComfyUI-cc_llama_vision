import base64
import hashlib
import io
import json
import os
import shlex
import shutil
import socket
import subprocess
import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import requests
from PIL import Image

# ---- Optional ComfyUI server integration (for the dynamic "refresh models" button) ----
try:
    from server import PromptServer
    from aiohttp import web
except ImportError:
    PromptServer = None
    web = None

# ---- Optional ComfyUI model-management integration (for VRAM unload node) ----
try:
    import comfy.model_management as comfy_mm
except ImportError:
    comfy_mm = None


# ---- Global server registry ----
# Key: (port, model_path, mmproj_path, ctx_size, n_gpu_layers, threads, threads_batch, extra_hash)
# Value: (subprocess.Popen, last_used_timestamp, file_handle)
_running_servers: Dict[Tuple, Tuple[subprocess.Popen, float, Optional[Any]]] = {}

# Default settings
DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser("~"), "models")
DEFAULT_LOG_PATH = os.path.join(
    os.path.expanduser("~"), "Documents", "ComfyUI", "llama_server_debug.log"
)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama_vision_config.json")

# Defaults for everything the "CC Llama Vision Advanced Options" node produces.
# Used both as that node's widget defaults and as the fallback the main node
# applies when no advanced_options input is connected, so the two stay in sync.
DEFAULT_ADVANCED_OPTS: Dict[str, Any] = {
    "n_gpu_layers": 99,
    "ctx_size": 8192,
    "threads": 0,
    "threads_batch": 0,
    "extra_server_args": "",
    "seed": 0,
    "temperature": 0.9,
    "top_p": 0.9,
    "top_k": 64,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "disable_thinking": True,
    "stop_sequences": "",
    "startup_timeout_s": 60,
    "request_timeout_s": 300,
    "keep_server_alive": False,
    "idle_timeout_s": 60,
    "force_restart": False,
    "debug": True,
    "server_log_path": DEFAULT_LOG_PATH,
}


# ---- Persistent config (remembers last-used models_dir across ComfyUI restarts) ----

def _load_config() -> dict:
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_config(update: dict) -> None:
    try:
        cfg = _load_config()
        cfg.update(update)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"[LlamaServer] Could not save config to {CONFIG_PATH}: {e}")


def _get_configured_models_dir() -> str:
    cfg = _load_config()
    d = cfg.get("models_dir")
    if d and os.path.isdir(d):
        return d
    return DEFAULT_MODELS_DIR


# ---- Helper functions ----

def _get_default_server_path() -> str:
    """
    Return the best default for llama-server.

    Preference order:
      1. Bare 'llama-server' if it resolves on PATH right now (e.g. installed/
         upgraded via `winget install llama.cpp` or similar). We deliberately
         keep it as the *bare command* (not the resolved absolute path) so
         that future winget upgrades which change the install location keep
         working without editing this field again.
      2. A legacy hardcoded install location, for users with an older manual
         install that isn't on PATH.
      3. Fall back to the bare command name and let the user fix PATH / the
         field if it's genuinely missing (validation gives a clear error).
    """
    if shutil.which("llama-server"):
        return "llama-server"
    legacy_path = r"C:\llama.cpp\llama-server.exe"
    if os.path.exists(legacy_path):
        return legacy_path
    return "llama-server"


def _scan_gguf_models(directory: str, preferred: Optional[str] = None,
                      name_filter: Optional[str] = None,
                      exclude_filter: Optional[str] = None) -> List[str]:
    """Recursively scan for .gguf files. Returns a sorted list."""
    found = []
    if directory and os.path.isdir(directory):
        for root, _, files in os.walk(directory):
            for fname in files:
                if not fname.lower().endswith(".gguf"):
                    continue
                lower = fname.lower()
                if name_filter and name_filter not in lower:
                    continue
                if exclude_filter and exclude_filter in lower:
                    continue
                found.append(os.path.join(root, fname))
    found.sort()
    if not found and preferred:
        found = [preferred]
    return found if found else [f"(no .gguf files found under {directory})"]


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_health(port: int, timeout_s: int, proc: subprocess.Popen) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    return False


def _tensor_to_data_url(image_tensor, format="JPEG") -> str:
    """Convert a ComfyUI image tensor to a data URL, using JPEG to reduce size."""
    img = image_tensor[0].cpu().numpy()
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    buf = io.BytesIO()
    pil_img.save(buf, format=format, quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64}"


def _kill_server_proc(proc: subprocess.Popen, timeout: int = 10, log_handle: Optional[Any] = None) -> None:
    if proc.poll() is None:
        proc.kill()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
    if log_handle is not None:
        try:
            log_handle.close()
        except Exception:
            pass


# ---- Dynamic models-dir API route ----
# Lets the frontend "Refresh Models" button rescan a folder the user just
# typed into the models_dir widget, without needing a full ComfyUI page
# reload. See js/llama_vision.js for the companion widget.
if PromptServer is not None and web is not None:

    @PromptServer.instance.routes.post("/llama_vision/scan_models")
    async def _llama_vision_scan_models(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        directory = (data or {}).get("dir") or DEFAULT_MODELS_DIR
        directory = os.path.expanduser(directory)

        if not os.path.isdir(directory):
            return web.json_response(
                {"error": f"Not a valid directory: {directory}",
                 "models": [f"(no .gguf files found under {directory})"],
                 "mmproj": [f"(no .gguf files found under {directory})"]},
                status=200,
            )

        models = _scan_gguf_models(directory, exclude_filter="mmproj")
        mmproj = _scan_gguf_models(directory, name_filter="mmproj")
        _save_config({"models_dir": directory})
        return web.json_response({"models": models, "mmproj": mmproj})


# ---- Main Node Class ----

class LlamaServerVisionCaption:
    @classmethod
    def INPUT_TYPES(cls):
        models_dir = _get_configured_models_dir()
        model_list = _scan_gguf_models(models_dir, None, exclude_filter="mmproj")
        mmproj_list = _scan_gguf_models(models_dir, None, name_filter="mmproj")

        return {
            "required": {
                "llama_server_path": ("STRING", {
                    "default": _get_default_server_path(),
                    "tooltip": "Path to the llama-server executable. Leave as 'llama-server' to "
                               "resolve automatically from PATH (recommended after "
                               "`winget install llama.cpp`), or provide a full path to "
                               "llama-server.exe.",
                }),
                "models_dir": ("STRING", {
                    "default": models_dir,
                    "tooltip": "Folder to scan (recursively) for .gguf model and mmproj files. "
                               "Click '🔄 Refresh Models' after changing this to repopulate the "
                               "dropdowns below.",
                }),
                "model_path": (model_list, {
                    "tooltip": "The GGUF vision-language model file to load "
                               "(mmproj files are excluded from this list).",
                }),
                "mmproj_path": (mmproj_list, {
                    "tooltip": "The GGUF multimodal projector (mmproj) file matching the selected "
                               "model — required for image/vision input.",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "System-level instructions sent to the model before the user "
                               "prompt, e.g. persona or output-format guidance.",
                }),
                "user_prompt": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "The instruction/question sent to the model along with the "
                               "image(s). This text is appended after any images.",
                }),
                "port": ("INT", {
                    "default": 8080, "min": 1024, "max": 65535,
                    "tooltip": "Local TCP port llama-server will listen on. Must be free unless "
                               "'keep_server_alive' is reusing an existing server on this port.",
                }),
                "max_tokens": ("INT", {
                    "default": 2048, "min": 16, "max": 8192,
                    "tooltip": "Maximum number of tokens the model may generate in its response.",
                }),
                "max_video_frames": ("INT", {
                    "default": 16, "min": 1, "max": 64,
                    "tooltip": "Maximum number of frames to sample from 'video_frames' input; "
                               "frames beyond this are evenly subsampled.",
                }),
                "label_video_frames": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, prefixes each video frame image with a text label "
                               "like '[Video frame N of M]'.",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "A single image to send to the model.",
                }),
                "image_batch": ("IMAGE", {
                    "tooltip": "A batch of images; each is sent as a separate image in the "
                               "same message.",
                }),
                "video_frames": ("IMAGE", {
                    "tooltip": "A sequence of image frames (e.g. from a video) to sample and "
                               "send to the model, subject to max_video_frames.",
                }),
                "advanced_options": ("LLAMA_VISION_OPTS", {
                    "tooltip": "Optional settings from a 'CC Llama Vision Advanced Options' "
                               "node — sampling, performance, lifecycle, and diagnostics. If "
                               "not connected, sensible defaults are used.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "run"
    CATEGORY = "llama.cpp"

    @classmethod
    def set_models_dir(cls, dir_path: str):
        """Kept for backward compatibility / scripted use. Prefer editing the
        models_dir widget + the 'Refresh Models' button, which persists to
        llama_vision_config.json automatically."""
        _save_config({"models_dir": dir_path})

    def _validate_file(self, path: str, desc: str) -> None:
        """Check if a file (model / mmproj) exists as an absolute/relative path."""
        if os.path.exists(path):
            return
        raise FileNotFoundError(
            f"{desc} not found: {path}. Double-check the 'models_dir' widget and the "
            "selected dropdown value, then click '🔄 Refresh Models' to rescan."
        )

    def _validate_executable(self, path: str, desc: str = "llama-server executable") -> None:
        """Check whether the llama-server executable can be located.

        Detection order (mirrors _get_default_server_path / _get_or_start_server):
          1. `path` exists directly on disk (absolute or relative file path).
          2. `path` resolves via shutil.which(), i.e. it is a bare command name
             (or a path) findable on the current process's PATH environment
             variable. This is what lets the 'llama-server' default keep working
             across winget upgrades that change the underlying install folder.

        If neither succeeds, raise a verbose, actionable error explaining how to
        install llama.cpp via winget on Windows, or how to add an existing
        install to PATH manually, since a missing/unresolvable executable is by
        far the most common setup problem for this node.
        """
        if os.path.exists(path):
            return
        if shutil.which(path) is not None:
            return

        raise FileNotFoundError(
            f"{desc} not found: '{path}'.\n\n"
            "llama-server could not be located either as a direct file path or "
            "anywhere on your system PATH. Here's how to fix it:\n\n"
            "== Option 1: Install/upgrade via winget (recommended on Windows) ==\n"
            "  1. Open a new PowerShell or Command Prompt window.\n"
            "  2. Run:\n"
            "       winget install llama.cpp\n"
            "     (already installed? upgrade instead with: winget upgrade llama.cpp)\n"
            "  3. Fully close and reopen ComfyUI (and any terminal windows) so the "
            "updated PATH is picked up by new processes.\n"
            "  4. Set the 'llama_server_path' widget to just:\n"
            "       llama-server\n"
            "     Leave off any folder — that way it keeps resolving via PATH even "
            "after future winget upgrades move the install location.\n\n"
            "== Option 2: Add an existing install to PATH manually (Windows) ==\n"
            "  1. Find the folder containing llama-server.exe "
            "(e.g. C:\\llama.cpp).\n"
            "  2. Press Win, search for 'Environment Variables', open "
            "'Edit the system environment variables'.\n"
            "  3. Click 'Environment Variables...' -> select 'Path' under "
            "'User variables' (or 'System variables') -> 'Edit...' -> 'New' -> "
            "paste the folder path -> OK on all open dialogs.\n"
            "  4. Or, from PowerShell (current user only, no admin needed):\n"
            "       setx PATH \"$env:PATH;C:\\llama.cpp\"\n"
            "  5. Fully close and reopen ComfyUI/terminal windows, then set "
            "'llama_server_path' to: llama-server\n\n"
            "== Option 3: Just use a full path ==\n"
            "  Skip PATH entirely and point 'llama_server_path' straight at the "
            "executable, e.g.:\n"
            "       C:\\llama.cpp\\llama-server.exe\n"
        )

    def _server_key(self, port: int, model_path: str, mmproj_path: str,
                    ctx_size: int, n_gpu_layers: int, extra_args: str,
                    threads: int, threads_batch: int) -> Tuple:
        """Generate a unique key for a server configuration."""
        extra_hash = hashlib.md5(extra_args.encode()).hexdigest() if extra_args else ""
        return (port, model_path, mmproj_path, ctx_size, n_gpu_layers,
                threads, threads_batch, extra_hash)

    def _get_or_start_server(self, llama_server_path: str, model_path: str,
                             mmproj_path: str, port: int, n_gpu_layers: int,
                             ctx_size: int, startup_timeout_s: int,
                             server_log_path: str, extra_server_args: str,
                             threads: int, threads_batch: int,
                             force_restart: bool, idle_timeout_s: int) -> subprocess.Popen:
        """Retrieve existing server if valid, or start a new one."""
        key = self._server_key(port, model_path, mmproj_path, ctx_size,
                               n_gpu_layers, extra_server_args, threads, threads_batch)

        # Force restart: kill any existing server with this key
        if force_restart and key in _running_servers:
            proc, _, log_f = _running_servers[key]
            if proc.poll() is None:
                print(f"[LlamaServer] Force restart: killing server on port {port}")
                _kill_server_proc(proc, timeout=5, log_handle=log_f)
            del _running_servers[key]
            # Give OS a moment to release the port
            time.sleep(0.2)

        # Check for existing server (now possibly removed)
        if key in _running_servers:
            proc, last_used, log_f = _running_servers[key]
            if proc.poll() is None:
                # Process alive – check idle timeout
                if idle_timeout_s > 0 and (time.time() - last_used) > idle_timeout_s:
                    print(f"[LlamaServer] Idle timeout ({idle_timeout_s}s) reached, killing server on port {port}")
                    _kill_server_proc(proc, timeout=5, log_handle=log_f)
                    del _running_servers[key]
                else:
                    # Valid and not idle – reuse
                    return proc
            else:
                # Process dead – remove from registry
                del _running_servers[key]

        # If port still in use, raise error (unless it's our just-killed process)
        if _port_in_use(port):
            raise RuntimeError(
                f"Port {port} is already in use. Please free it or choose a different port."
            )

        # Start new server
        self._validate_executable(llama_server_path)
        exe_path = shutil.which(llama_server_path) if not os.path.exists(llama_server_path) else llama_server_path

        cmd = [
            exe_path,
            "-m", model_path,
            "--mmproj", mmproj_path,
            "--port", str(port),
            "-ngl", str(n_gpu_layers),
            "--ctx-size", str(ctx_size),
            "--jinja",
        ]
        if threads > 0:
            cmd += ["--threads", str(threads)]
        if threads_batch > 0:
            cmd += ["--threads-batch", str(threads_batch)]
        if extra_server_args.strip():
            try:
                cmd.extend(shlex.split(extra_server_args, posix=os.name != "nt"))
            except ValueError as exc:
                raise ValueError(f"Could not parse extra_server_args: {exc}") from exc

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NO_WINDOW

        log_dir = os.path.dirname(server_log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_f = open(server_log_path, "w", encoding="utf-8", errors="replace")

        try:
            print(f"[LlamaServer] Starting server on port {port}...")
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )

            if not _wait_for_health(port, startup_timeout_s, proc):
                if proc.poll() is not None:
                    log_f.close()
                    raise RuntimeError(
                        f"llama-server exited early (code {proc.poll()}). Check log: {server_log_path}"
                    )
                else:
                    log_f.close()
                    raise RuntimeError(
                        f"llama-server did not become healthy within {startup_timeout_s}s. Check log: {server_log_path}"
                    )

            print(f"[LlamaServer] Server ready on port {port}")
            _running_servers[key] = (proc, time.time(), log_f)
            return proc
        except Exception:
            log_f.close()
            raise

    def _update_last_used(self, key: Tuple):
        if key in _running_servers:
            proc, _, log_f = _running_servers[key]
            _running_servers[key] = (proc, time.time(), log_f)

    def run(self, llama_server_path, models_dir, model_path, mmproj_path,
            system_prompt, user_prompt, port, max_tokens,
            max_video_frames, label_video_frames,
            image=None, image_batch=None, video_frames=None,
            advanced_options=None):

        # Fill in anything not supplied by an "Advanced Options" node.
        opts = dict(DEFAULT_ADVANCED_OPTS)
        if advanced_options:
            opts.update(advanced_options)

        n_gpu_layers = opts["n_gpu_layers"]
        ctx_size = opts["ctx_size"]
        threads = opts["threads"]
        threads_batch = opts["threads_batch"]
        extra_server_args = opts["extra_server_args"]
        seed = opts["seed"]
        temperature = opts["temperature"]
        top_p = opts["top_p"]
        top_k = opts["top_k"]
        min_p = opts["min_p"]
        repeat_penalty = opts["repeat_penalty"]
        presence_penalty = opts["presence_penalty"]
        frequency_penalty = opts["frequency_penalty"]
        disable_thinking = opts["disable_thinking"]
        stop_sequences = opts["stop_sequences"]
        startup_timeout_s = opts["startup_timeout_s"]
        request_timeout_s = opts["request_timeout_s"]
        keep_server_alive = opts["keep_server_alive"]
        idle_timeout_s = opts["idle_timeout_s"]
        force_restart = opts["force_restart"]
        debug = opts["debug"]
        server_log_path = opts["server_log_path"]

        # Persist the models_dir the user is actually using, so it's
        # remembered as the default next time ComfyUI (re)loads this node.
        if models_dir and os.path.isdir(models_dir):
            _save_config({"models_dir": models_dir})

        # Validate files
        self._validate_executable(llama_server_path, "llama-server executable")
        self._validate_file(model_path, "Model file")
        self._validate_file(mmproj_path, "mmproj file")

        proc = None
        server_key = None
        try:
            if keep_server_alive:
                proc = self._get_or_start_server(
                    llama_server_path, model_path, mmproj_path, port,
                    n_gpu_layers, ctx_size, startup_timeout_s,
                    server_log_path, extra_server_args, threads, threads_batch,
                    force_restart, idle_timeout_s
                )
                server_key = self._server_key(port, model_path, mmproj_path, ctx_size,
                                              n_gpu_layers, extra_server_args, threads, threads_batch)
            else:
                # Start a temporary server (will be killed after request)
                if _port_in_use(port):
                    raise RuntimeError(
                        f"Port {port} is already in use. Either free it, choose another port, "
                        "or enable 'keep_server_alive' to reuse an existing server."
                    )
                proc = self._get_or_start_server(
                    llama_server_path, model_path, mmproj_path, port,
                    n_gpu_layers, ctx_size, startup_timeout_s,
                    server_log_path, extra_server_args, threads, threads_batch,
                    force_restart=False, idle_timeout_s=0  # no auto-unload for temp
                )

            # ---- Build multimodal content ----
            content_items = []
            if image is not None:
                content_items.append(("image", image[0:1]))
            if image_batch is not None:
                for i in range(image_batch.shape[0]):
                    content_items.append(("image", image_batch[i:i+1]))
            if video_frames is not None:
                total = video_frames.shape[0]
                if total > max_video_frames:
                    idxs = sorted(set(
                        np.linspace(0, total - 1, max_video_frames).round().astype(int).tolist()
                    ))
                else:
                    idxs = list(range(total))
                for idx in idxs:
                    if label_video_frames:
                        content_items.append(("text", f"[Video frame {idx+1} of {total}]"))
                    content_items.append(("image", video_frames[idx:idx+1]))

            user_content = []
            if content_items:
                for kind, payload in content_items:
                    if kind == "image":
                        data_url = _tensor_to_data_url(payload, format="JPEG")
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": data_url}
                        })
                    else:
                        user_content.append({"type": "text", "text": payload})
                user_content.append({"type": "text", "text": user_prompt})
            else:
                user_content = user_prompt

            # ---- Prepare API payload ----
            payload = {
                "model": "vision-caption",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": max_tokens,
                "seed": seed,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "repeat_penalty": repeat_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
            }

            stops = [s.strip() for s in stop_sequences.splitlines() if s.strip()]
            if stops:
                payload["stop"] = stops

            if disable_thinking:
                payload["chat_template_kwargs"] = {"thinking": False}

            # ---- Send request ----
            resp = requests.post(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                json=payload,
                timeout=request_timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()

            if debug:
                print("[LlamaServerVisionCaption] Raw response:")
                print(json.dumps(data, indent=2)[:4000])

            message = data["choices"][0]["message"]
            finish_reason = data["choices"][0].get("finish_reason")
            caption = message.get("content") or ""

            if not caption.strip():
                reasoning = message.get("reasoning_content") or ""
                if reasoning.strip():
                    caption = (
                        f"[empty 'content' field — model likely still reasoning "
                        f"when max_tokens was hit (finish_reason={finish_reason}). "
                        f"reasoning_content was:]\n\n{reasoning}"
                    )
                else:
                    caption = (
                        f"[no text returned — finish_reason={finish_reason}. "
                        f"Check {server_log_path} for template/vision errors. "
                        f"Raw message: {json.dumps(message)[:1000]}]"
                    )

            # Update last used time if persistent server
            if keep_server_alive and server_key:
                self._update_last_used(server_key)

            return (caption,)

        finally:
            # If not keeping alive, kill the temporary server
            if not keep_server_alive and proc is not None:
                _kill_server_proc(proc, timeout=10)
                # Remove from registry if present (shouldn't be)
                for key, (p, _, log_f) in list(_running_servers.items()):
                    if p == proc:
                        del _running_servers[key]
                        break
                print("[LlamaServer] Temporary server stopped.")


# ---- Advanced Options node ----
# Feeds a bundle of sampling / performance / lifecycle / diagnostics settings
# into LlamaServerVisionCaption's optional "advanced_options" input, so the
# main node's widget list stays focused on model selection, prompts, and
# media. Leaving this node disconnected is equivalent to using the defaults
# below (kept in sync via DEFAULT_ADVANCED_OPTS).
class LlamaVisionAdvancedOptions:
    @classmethod
    def INPUT_TYPES(cls):
        d = DEFAULT_ADVANCED_OPTS
        return {
            "required": {
                "temperature": ("FLOAT", {
                    "default": d["temperature"], "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Sampling temperature. Higher = more random/creative, lower = "
                               "more deterministic/focused.",
                }),
                "top_p": ("FLOAT", {
                    "default": d["top_p"], "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Nucleus sampling threshold — only tokens within this cumulative "
                               "probability mass are considered.",
                }),
                "top_k": ("INT", {
                    "default": d["top_k"], "min": 0, "max": 1000,
                    "tooltip": "Only the top K most likely tokens are considered at each step. "
                               "0 disables this filter.",
                }),
                "min_p": ("FLOAT", {
                    "default": d["min_p"], "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Minimum probability (relative to the top token) a token must "
                               "have to be considered.",
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": d["repeat_penalty"], "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Penalty applied to tokens that have already appeared, to "
                               "discourage repetition.",
                }),
                "presence_penalty": ("FLOAT", {
                    "default": d["presence_penalty"], "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Penalty applied per unique token already present in the output, "
                               "encouraging new topics.",
                }),
                "frequency_penalty": ("FLOAT", {
                    "default": d["frequency_penalty"], "min": -2.0, "max": 2.0, "step": 0.01,
                    "tooltip": "Penalty scaled by how often a token has already appeared, "
                               "discouraging repetition.",
                }),
                "seed": ("INT", {"default": d["seed"], "min": -1, "max": 0xFFFFFFFFFFFFFFFF,
                                  "control_after_generate": "fixed",
                                  "tooltip": "Random seed for generation. -1 for random each run; "
                                            "a fixed value makes output reproducible."}),
                "disable_thinking": ("BOOLEAN", {
                    "default": d["disable_thinking"],
                    "tooltip": "If enabled, disables the model's internal 'thinking'/reasoning "
                               "mode (for models that support it) so it responds directly.",
                }),
                "stop_sequences": ("STRING", {
                    "multiline": True, "default": d["stop_sequences"],
                    "tooltip": "One stop string per line; generation halts early if any of "
                               "these strings are produced.",
                }),
                "n_gpu_layers": ("INT", {
                    "default": d["n_gpu_layers"], "min": 0, "max": 200,
                    "tooltip": "Number of model layers to offload to GPU. Higher = faster but "
                               "more VRAM; set to 0 for CPU-only.",
                }),
                "ctx_size": ("INT", {
                    "default": d["ctx_size"], "min": 512, "max": 131072,
                    "tooltip": "Context window size (tokens) for the server. Larger allows "
                               "longer prompts/images but uses more VRAM/RAM.",
                }),
                "threads": ("INT", {
                    "default": d["threads"], "min": 0, "max": 256, "label": "CPU threads (0=auto)",
                    "tooltip": "Number of CPU threads for generation. 0 = let llama-server "
                               "auto-detect.",
                }),
                "threads_batch": ("INT", {
                    "default": d["threads_batch"], "min": 0, "max": 256,
                    "label": "Batch threads (0=auto)",
                    "tooltip": "Number of CPU threads for batch/prompt processing. 0 = let "
                               "llama-server auto-detect.",
                }),
                "extra_server_args": ("STRING", {
                    "default": d["extra_server_args"],
                    "tooltip": "Additional raw command-line arguments passed through to "
                               "llama-server (advanced use).",
                }),
                "keep_server_alive": ("BOOLEAN", {
                    "default": d["keep_server_alive"], "label": "Keep server alive (reuse across runs)",
                    "tooltip": "Keep llama-server running after this node finishes so future "
                               "runs can reuse it instead of restarting (faster, but keeps "
                               "VRAM occupied).",
                }),
                "idle_timeout_s": ("INT", {
                    "default": d["idle_timeout_s"], "min": 0, "max": 3600,
                    "label": "Auto-unload idle timeout (0=never)",
                    "tooltip": "If a kept-alive server goes unused for this many seconds, it "
                               "will be automatically killed to free VRAM. 0 disables "
                               "auto-unload.",
                }),
                "force_restart": ("BOOLEAN", {
                    "default": d["force_restart"], "label": "Force restart server (ignore existing)",
                    "tooltip": "Kill and restart any matching existing server before running, "
                               "even if one is already alive and healthy.",
                }),
                "startup_timeout_s": ("INT", {
                    "default": d["startup_timeout_s"], "min": 5, "max": 300,
                    "tooltip": "How many seconds to wait for llama-server to report healthy "
                               "before giving up on startup.",
                }),
                "request_timeout_s": ("INT", {
                    "default": d["request_timeout_s"], "min": 30, "max": 1800,
                    "tooltip": "How many seconds to wait for a response to the captioning "
                               "request before timing out.",
                }),
                "debug": ("BOOLEAN", {
                    "default": d["debug"],
                    "tooltip": "Print the raw JSON response from llama-server to the console "
                               "for troubleshooting.",
                }),
                "server_log_path": ("STRING", {
                    "default": d["server_log_path"],
                    "tooltip": "File path where llama-server's stdout/stderr log will be "
                               "written — check this file if startup fails.",
                }),
            },
        }

    RETURN_TYPES = ("LLAMA_VISION_OPTS",)
    RETURN_NAMES = ("advanced_options",)
    FUNCTION = "build"
    CATEGORY = "llama.cpp"

    def build(self, temperature, top_p, top_k, min_p, repeat_penalty,
              presence_penalty, frequency_penalty, seed, disable_thinking,
              stop_sequences, n_gpu_layers, ctx_size, threads, threads_batch,
              extra_server_args, keep_server_alive, idle_timeout_s,
              force_restart, startup_timeout_s, request_timeout_s, debug,
              server_log_path):
        return ({
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repeat_penalty": repeat_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "seed": seed,
            "disable_thinking": disable_thinking,
            "stop_sequences": stop_sequences,
            "n_gpu_layers": n_gpu_layers,
            "ctx_size": ctx_size,
            "threads": threads,
            "threads_batch": threads_batch,
            "extra_server_args": extra_server_args,
            "keep_server_alive": keep_server_alive,
            "idle_timeout_s": idle_timeout_s,
            "force_restart": force_restart,
            "startup_timeout_s": startup_timeout_s,
            "request_timeout_s": request_timeout_s,
            "debug": debug,
            "server_log_path": server_log_path,
        },)


# ---- VRAM handoff node ----
# llama-server runs as an external subprocess, so ComfyUI's own model
# manager has no idea it is holding VRAM. If a workflow does
# "caption with the LLM" -> "generate an image with a big diffusion model"
# in two separate steps, the second step can OOM because the LLM never got
# unloaded. Chain this node between the caption node and whatever consumes
# its text (e.g. a CLIPTextEncode "text" input) — since ComfyUI executes in
# dependency order, that link *guarantees* the llama-server is killed before
# the image-model node runs, without needing timers or guesswork.
class LlamaServerUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Upstream string output to chain from (e.g. the caption output) "
                               "— guarantees this node runs before whatever consumes it, since "
                               "ComfyUI executes in dependency order.",
                }),
                "target": (["all_servers", "specific_port"], {
                    "default": "all_servers",
                    "tooltip": "Which server(s) to kill: every currently tracked llama-server "
                               "process, or only the one on the specified port.",
                }),
                "also_free_comfyui_vram": ("BOOLEAN", {
                    "default": True,
                    "label": "Also unload ComfyUI's own models / empty cache",
                    "tooltip": "Also ask ComfyUI to unload its own loaded models and empty its "
                               "VRAM cache after killing llama-server.",
                }),
            },
            "optional": {
                "port": ("INT", {
                    "default": 8080, "min": 1024, "max": 65535,
                    "tooltip": "Port of the specific llama-server to kill, used only when "
                               "'target' is 'specific_port'.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("trigger",)
    FUNCTION = "unload"
    CATEGORY = "llama.cpp"
    OUTPUT_NODE = False

    def unload(self, trigger, target, also_free_comfyui_vram, port=8080):
        killed_ports = []
        for key, (proc, _, log_f) in list(_running_servers.items()):
            key_port = key[0]
            if target == "all_servers" or key_port == port:
                _kill_server_proc(proc, timeout=10, log_handle=log_f)
                del _running_servers[key]
                killed_ports.append(key_port)

        if killed_ports:
            print(f"[LlamaServerUnload] Killed llama-server(s) on port(s): {killed_ports}")
        else:
            print("[LlamaServerUnload] No matching llama-server was running.")

        if also_free_comfyui_vram and comfy_mm is not None:
            try:
                comfy_mm.unload_all_models()
                comfy_mm.soft_empty_cache()
                print("[LlamaServerUnload] Freed ComfyUI-managed VRAM cache.")
            except Exception as e:
                print(f"[LlamaServerUnload] Could not free ComfyUI VRAM cache: {e}")

        return (trigger,)


# ---- ComfyUI Node Registration ----
NODE_CLASS_MAPPINGS = {
    "LlamaServerVisionCaption": LlamaServerVisionCaption,
    "LlamaVisionAdvancedOptions": LlamaVisionAdvancedOptions,
    "LlamaServerUnload": LlamaServerUnload,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LlamaServerVisionCaption": "CC Llama Vision",
    "LlamaVisionAdvancedOptions": "CC Llama Vision Advanced Options",
    "LlamaServerUnload": "CC Llama Server Unload (free VRAM)",
}
