// ComfyUI-Noctyra — 动态控制参数显隐
// GPL-3.0 (见仓库 LICENSE)
//
// 按"控制 widget"(下拉/开关/数值)的值，显示或隐藏相关的控制参数。
// 机制参考 comfyui-easy-use 的 easyDynamicWidgets：隐藏=把 widget.type 改成
// 不渲染的类型并令其高度为 0；给控制 widget 的 value 装 getter/setter，值一变
// 就重新计算显隐。纯前端，不改 Python 节点定义。

// 新前端(1.42+)用全局 window.comfyAPI 取 app（与 kjnodes 同款，ES import 在新前端已失效）。
const { app } = window.comfyAPI.app;

console.log("[Noctyra] 动态参数显隐扩展已加载");

// ── 配置表：节点类名 -> 规则数组 ─────────────────────────────────────
// 每条规则：controller=控制 widget 名；managed=该规则掌管的 widget；
// show(value)=返回当前值下应"显示"的 widget 名数组(其余 managed 的隐藏)。
const NODE_RULES = {
    // 去除可见水印（区域修复）：按"模式"显示对应区域参数；按"后端"显隐 cv2 参数
    "RemoveVisibleWatermark": [
        {
            controller: "模式",
            managed: ["角落位置", "角落大小", "条带位置", "条带高度",
                      "自定义X", "自定义Y", "自定义宽", "自定义高"],
            show: (v) => ({
                "角落": ["角落位置", "角落大小"],
                "条带": ["条带位置", "条带高度"],
                "自定义矩形": ["自定义X", "自定义Y", "自定义宽", "自定义高"],
                "外部遮罩": [],
            }[v] || []),
        },
        {
            controller: "后端",
            managed: ["修复方法", "修复半径"],
            // cv2 后端才用得到 修复方法/半径；LaMa 不用
            show: (v) => String(v).startsWith("cv2") ? ["修复方法", "修复半径"] : [],
        },
    ],
    // 去除 Gemini 星标水印：按「去除方式」显隐对应参数
    "RemoveGeminiWatermark": [
        {
            controller: "去除方式",
            managed: ["模板尺寸", "修复区域扩大", "残留修复", "修复强度"],
            // 直接修复：显示「修复区域扩大」；反向还原：显示模板尺寸/残留修复/修复强度
            show: (v) => String(v).startsWith("直接")
                ? ["修复区域扩大"]
                : ["模板尺寸", "残留修复", "修复强度"],
        },
    ],
    // 生成文字水印：描边宽度>0 才显示描边颜色
    "CreateTextWatermark": [
        {
            controller: "描边宽度",
            managed: ["描边颜色"],
            show: (v) => Number(v) > 0 ? ["描边颜色"] : [],
        },
    ],
    // 隐形水印模型下载器：管线含 SDXL(default/全部) 才显示「自定义SDXL模型」
    "InvisibleWatermarkModelDownloader": [
        {
            controller: "管线",
            managed: ["自定义SDXL模型"],
            show: (v) => (String(v).startsWith("default") || v === "全部") ? ["自定义SDXL模型"] : [],
        },
    ],
};

// ── 显隐一个 widget ───────────────────────────────────────────────────
// 从前端源码确认，新前端(1.42)判断隐藏分两处：
//   · 内容渲染(Vue)：读 widget.options.hidden
//   · 布局/高度    ：读 widget.hidden —— 它是只读 getter，转读"最深层"widget
//     的 hidden(resolveDeepest().widget.hidden)。所以必须把 hidden 设到最深层，
//     否则内容隐藏了但布局没折叠(留空隙、还能点出弹框)。
//   · 经典 litegraph：computeSize 归零 + 未知 type。
function toggleWidget(node, widget, show) {
    if (!widget) return;
    if (!widget._noctyraOrig) {
        widget._noctyraOrig = { type: widget.type, computeSize: widget.computeSize };
    }
    const hide = !show;

    // 设到最深层 widget（新前端 hidden getter 实际读这里）
    const inner = (typeof widget.resolveDeepest === "function" && widget.resolveDeepest()?.widget) || widget;
    for (const w of new Set([widget, inner])) {
        try { w.hidden = hide; } catch (e) { /* 只读 getter 则忽略 */ }
        if (!w.options) w.options = {};
        w.options.hidden = hide;
    }

    // 经典 litegraph：高度归零 + 未知 type
    widget.type = show ? widget._noctyraOrig.type : "noctyra-hidden";
    widget.computeSize = show ? widget._noctyraOrig.computeSize : () => [0, -4];
}

function findWidget(node, name) {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}

// 当前是否 Nodes 2.0 (Vue) 渲染
function isVueNodes() {
    try { return !!app.ui?.settings?.getSettingValue?.("Comfy.VueNodes.Enabled"); }
    catch (e) { return false; }
}

// Nodes 2.0 下，改 options.hidden 不会让已挂载的 Vue 节点组件重渲（实测：唯一可靠
// 的办法是让该节点的组件重新挂载，即把节点 remove 再 add）。这里 remove+add 并
// 保存/恢复其输入输出连接，避免断线；用 _noctyraRefreshing 守卫防止 add 重新触发
// nodeCreated 造成递归。
function forceRerender(node) {
    const graph = node.graph || app.graph;
    if (!graph || node._noctyraRefreshing) return;
    // 只在节点已加入图中时才 remove+add；创建阶段(nodeCreated 早于 graph.add)若
    // 强行 add 会与随后的正常 add 造成"重复节点"。此时无需重渲——首次挂载本就读对。
    if (!Array.isArray(graph._nodes) || !graph._nodes.includes(node)) return;
    node._noctyraRefreshing = true;
    try {
        const inLinks = (node.inputs || []).map((inp, i) => {
            const lk = inp && inp.link != null ? graph.links[inp.link] : null;
            return lk ? { slot: i, origin_id: lk.origin_id, origin_slot: lk.origin_slot } : null;
        }).filter(Boolean);
        const outLinks = [];
        (node.outputs || []).forEach((out, i) => {
            (out.links || []).slice().forEach((lid) => {
                const lk = graph.links[lid];
                if (lk) outLinks.push({ slot: i, target_id: lk.target_id, target_slot: lk.target_slot });
            });
        });
        const pos = node.pos ? [node.pos[0], node.pos[1]] : null;
        const width = node.size ? node.size[0] : null;   // 仅保留宽度，高度自适应

        graph.remove(node);
        graph.add(node);
        if (pos) node.pos = pos;
        // 高度按当前可见 widget 重算（隐藏的 computeSize 已归零），宽度沿用
        try {
            const w = width != null ? width : node.size[0];
            node.setSize([w, node.computeSize()[1]]);
        } catch (e) {
            if (width != null && node.size) node.size[0] = width;
        }

        for (const s of inLinks) {
            const src = graph.getNodeById(s.origin_id);
            if (src) src.connect(s.origin_slot, node, s.slot);
        }
        for (const s of outLinks) {
            const tgt = graph.getNodeById(s.target_id);
            if (tgt) node.connect(s.slot, tgt, s.target_slot);
        }
    } catch (e) {
        console.warn("[Noctyra] 强制重渲失败:", e);
    } finally {
        node._noctyraRefreshing = false;
    }
}

// ── 对一个节点应用全部规则 ───────────────────────────────────────────
function applyRules(node, rules) {
    let n = 0;
    for (const rule of rules) {
        const ctrl = findWidget(node, rule.controller);
        if (!ctrl) continue;
        const showSet = new Set(rule.show(ctrl.value));
        for (const name of rule.managed) {
            const w = findWidget(node, name);
            if (w) { toggleWidget(node, w, showSet.has(name)); n++; }
        }
    }
    // 重渲由调用方在 nodeCreated 重入时跳过（_noctyraRefreshing）
    if (n > 0 && !node._noctyraRefreshing) {
        if (isVueNodes()) {
            forceRerender(node);            // Nodes 2.0：重新挂载组件
        } else {
            try {                           // 经典 litegraph：重算高度即可
                const sz = node.computeSize();
                node.setSize([node.size[0], sz[1]]);
            } catch (e) {}
        }
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
        app.canvas?.setDirty?.(true, true);
    }
}

// ── 给控制 widget 同时挂 callback + value setter，确保两种渲染器都能触发 ──
function hookControllers(node, rules) {
    for (const rule of rules) {
        const ctrl = findWidget(node, rule.controller);
        if (!ctrl || ctrl._noctyraHooked) continue;
        ctrl._noctyraHooked = true;

        // 1) 包 callback（用户改值时，经典/Vue 渲染器都会调用它）
        const cb = ctrl.callback;
        ctrl.callback = function () {
            const r = cb ? cb.apply(this, arguments) : undefined;
            applyRules(node, rules);
            return r;
        };
        // 2) 再装 value getter/setter（捕获程序化赋值；失败忽略）
        try {
            let val = ctrl.value;
            Object.defineProperty(ctrl, "value", {
                get() { return val; },
                set(nv) { const d = nv !== val; val = nv; if (d) applyRules(node, rules); },
                configurable: true,
            });
        } catch (e) { /* 不可重定义则仅靠 callback */ }
    }
}

function setup(node, rules) {
    // 等 widgets 就绪再执行（最多重试几帧）
    let tries = 0;
    const run = () => {
        if (!node.widgets || node.widgets.length === 0) {
            if (tries++ < 10) return setTimeout(run, 50);
            return;
        }
        try {
            hookControllers(node, rules);
            applyRules(node, rules);
        } catch (e) { console.warn("[Noctyra] 动态显隐失败(不影响功能):", e); }
    };
    run();
}

app.registerExtension({
    name: "Noctyra.DynamicWidgets",
    nodeCreated(node) {
        const rules = NODE_RULES[node.comfyClass || node.type];
        if (!rules) return;
        // forceRerender 的 remove+add 会重触发本钩子；此时 options.hidden 已设好，
        // 直接跳过，避免嵌套 setup/applyRules 与挂载时机打架。
        if (node._noctyraRefreshing) return;
        setup(node, rules);
        const onConfigure = node.onConfigure;
        node.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            setTimeout(() => { try { applyRules(node, rules); } catch (e) {} }, 0);
        };
    },
});

// ── 互斥输入：同组输入只能连一个，二选一 ─────────────────────────────────
// 配置：节点类名 -> 互斥组数组，每组是一批互斥的输入名。
const EXCLUSIVE_INPUTS = {
    // AI水印识别：「图像」与「图片路径」二选一
    "IdentifyAIProvenance": [["图像", "图片路径"]],
};

function inputIndex(node, name) {
    return (node.inputs || []).findIndex((i) => i.name === name);
}
function isInputConnected(node, name) {
    const i = inputIndex(node, name);
    return i >= 0 && node.inputs[i].link != null;
}

// 轻提示（新前端有 toast 则用，否则退到控制台）
function notify(msg) {
    try {
        const t = app.extensionManager?.toast;
        if (t?.add) { t.add({ severity: "warn", summary: "Noctyra", detail: msg, life: 3000 }); return; }
    } catch (e) {}
    console.warn("[Noctyra] " + msg);
}

app.registerExtension({
    name: "Noctyra.ExclusiveInputs",
    nodeCreated(node) {
        const groups = EXCLUSIVE_INPUTS[node.comfyClass || node.type];
        if (!groups) return;

        // onConnectInput 在连接「建立之前」调用，返回 false 即拒绝——比事后断开更干净。
        const origConnectInput = node.onConnectInput;
        node.onConnectInput = function (targetSlot, type, output, originNode, originSlot) {
            const r = origConnectInput ? origConnectInput.apply(this, arguments) : true;
            if (r === false) return false;
            try {
                const slot = this.inputs?.[targetSlot];
                if (slot) {
                    for (const grp of groups) {
                        if (!grp.includes(slot.name)) continue;
                        const otherName = grp.find((o) => o !== slot.name && isInputConnected(this, o));
                        if (otherName) {
                            notify(`「${slot.name}」与「${otherName}」二选一，请先断开「${otherName}」`);
                            return false;   // 另一个已连接 → 拒绝本次连接
                        }
                    }
                }
            } catch (e) { console.warn("[Noctyra] 互斥输入判断失败:", e); }
            return true;
        };
    },
});
