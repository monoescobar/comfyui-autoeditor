import { app } from "../../scripts/app.js";

const AUTO_EDITOR_NODE_NAMES = new Set(["DJ_AutoEditor", "DJ_AutoDirector"]);
const VIDEO_UNDERSTANDING_VALUES = new Set(["ON", "FAST", "OFF"]);

function normalizedDuration(value) {
    const text = String(value ?? "").trim();
    if (!text) return null;
    const parsed = Number(text);
    if (!Number.isFinite(parsed) || parsed < 0) return null;
    return String(Math.round(parsed));
}

function migrateAutoEditorWidgets(node) {
    const widgets = node.widgets ?? [];
    const videoWidget = widgets.find((widget) => widget.name === "video_understanding");
    const durationWidget = widgets.find((widget) => widget.name === "target_duration_seconds");
    if (!videoWidget || !durationWidget) return false;

    let changed = false;
    const videoValue = String(videoWidget.value ?? "").trim().toUpperCase();
    const durationValue = String(durationWidget.value ?? "").trim();

    if (!VIDEO_UNDERSTANDING_VALUES.has(videoValue)) {
        const misplacedDuration = normalizedDuration(videoWidget.value);
        if (
            misplacedDuration !== null
            && misplacedDuration !== "0"
            && (durationValue === "" || durationValue === "0")
        ) {
            durationWidget.value = misplacedDuration;
        }
        videoWidget.value = "FAST";
        changed = true;
    }

    const cleanDuration = normalizedDuration(durationWidget.value);
    if (cleanDuration === null) {
        durationWidget.value = "0";
        changed = true;
    } else if (String(durationWidget.value) !== cleanDuration) {
        durationWidget.value = cleanDuration;
        changed = true;
    }

    if (changed) {
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    }
    return changed;
}

app.registerExtension({
    name: "comfyui-autoeditor.widget-migration",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!AUTO_EDITOR_NODE_NAMES.has(nodeData.name)) return;

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            migrateAutoEditorWidgets(this);
            requestAnimationFrame(() => migrateAutoEditorWidgets(this));
            return result;
        };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            requestAnimationFrame(() => migrateAutoEditorWidgets(this));
            return result;
        };
    },
});