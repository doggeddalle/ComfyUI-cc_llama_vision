from .cc_llama_vision import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Tells ComfyUI to serve ./js/*.js to the frontend so the "Refresh Models"
# button widget in llama_vision.js gets loaded.
WEB_DIRECTORY = "js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
