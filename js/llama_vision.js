import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Adds a "Refresh Models" button to the CC Llama Vision node. Clicking it
// (re)scans whatever folder is currently typed into the models_dir widget
// and repopulates the model_path / mmproj_path dropdowns in place, no
// ComfyUI page reload required. The scanned folder is also remembered
// server-side (llama_vision_config.json) as the default for next time.
app.registerExtension({
    name: "ccLlamaVision.refreshModels",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LlamaServerVisionCaption") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const modelsDirWidget = this.widgets?.find((w) => w.name === "models_dir");
            const modelWidget = this.widgets?.find((w) => w.name === "model_path");
            const mmprojWidget = this.widgets?.find((w) => w.name === "mmproj_path");

            const setComboValues = (widget, values) => {
                if (!widget || !values?.length) return;
                widget.options = widget.options || {};
                const current = widget.value;
                const isPlaceholder =
                    typeof current !== "string" || !current || current.startsWith("(no .gguf");
                if (!values.includes(current) && !isPlaceholder) {
                    // Keep a saved selection that isn't in the rescanned list
                    // (e.g. a workflow from another machine) instead of
                    // silently switching models; the backend reports a clear
                    // error at run time if the file really is missing.
                    values = [current, ...values];
                }
                widget.options.values = values;
                if (!values.includes(widget.value)) {
                    widget.value = values[0];
                }
            };

            const refresh = async () => {
                const dir = modelsDirWidget ? modelsDirWidget.value : "";
                try {
                    const resp = await api.fetchApi("/llama_vision/scan_models", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ dir }),
                    });
                    const data = await resp.json();
                    if (data.error) {
                        console.warn("[CC Llama Vision]", data.error);
                    }
                    setComboValues(modelWidget, data.models);
                    setComboValues(mmprojWidget, data.mmproj);
                    this.setDirtyCanvas(true, true);
                } catch (e) {
                    console.error("[CC Llama Vision] Failed to refresh models:", e);
                }
            };

            this.addWidget("button", "🔄 Refresh Models", null, refresh);

            // Also refresh once automatically on node creation so a freshly
            // dropped node picks up whatever folder is already configured.
            refresh();

            return result;
        };
    },
});
