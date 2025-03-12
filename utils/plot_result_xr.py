import numpy as np
import matplotlib.pyplot as plt

datasets = {
    "IBD": {
        "Methods": ["CNN+MLP_C", "CNN+MLP_F", "CNN+MLP_P", "CNN+MLP_O", "MLP_C", "MLP_F", "MLP_P", "MLP_O"],
        "ALL_Scc": [0.3739, 0.3665, 0.3568, 0.3482, 0.3624, 0.3566, 0.3579, 0.3319],
        "WP_Nums": [5711, 5572, 5412, 5385, 5538, 5478, 5441, 5197]
    },
    "ESRD": {
        "Methods": ["CNN+MLP_C", "CNN+MLP_F", "CNN+MLP_P", "CNN+MLP_O", "MLP_C", "MLP_F", "MLP_P", "MLP_O"],
        "ALL_Scc": [0.2478, 0.2373, 0.2257, 0.2179, 0.2313, 0.2167, 0.2218, 0.2141],
        "WP_Nums": [124, 109, 92, 86, 97, 80, 86, 79]
    },
    "GC": {
        "Methods": ["CNN+MLP_C", "CNN+MLP_F", "CNN+MLP_P", "CNN+MLP_O", "MLP_C", "MLP_F", "MLP_P", "MLP_O"],
        "ALL_Scc": [0.2513, 0.2172, 0.1939, 0.1771, 0.1493, 0.1452, 0.1519, 0.1109],
        "WP_Nums": [269, 222, 112, 97, 92, 83, 84, 81]
    },
    "CRC": {
        "Methods": ["CNN+MLP_C", "CNN+MLP_F", "CNN+MLP_P", "CNN+MLP_O", "MLP_C", "MLP_F", "MLP_P", "MLP_O"],
        "ALL_Scc": [0.2399, 0.2035, 0.2159, 0.1641, 0.1531, 0.1314, 0.1384, 0.1223],
        "WP_Nums": [115, 94, 97, 67, 71, 61, 65, 45]
    }
}

# ✅ 颜色 & 样式
bar_width = 0.4
colors = {"ALL_Scc": "#1f77b4", "WP_Nums": "#ff7f0e"}  # 蓝色 & 橙色
hatch_style = "/"
label_fontsize = 14
tick_fontsize = 12

# ✅ 创建 2×2 旋转子图（更紧凑）
fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

# ✅ 添加全局颜色示例（单独的图例框架）【图例放在 Figure 之外，避免和子图重叠】
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, fc=colors["ALL_Scc"], label="ALL_Scc"),
    plt.Rectangle((0, 0), 1, 1, fc=colors["WP_Nums"], hatch=hatch_style, label="WP_Nums")
]
fig.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.53, 1.03), ncol=2, fontsize=label_fontsize, frameon=False)

# ✅ 遍历数据集并绘制（添加子图标签 (a), (b), (c), (d)）
for idx, (dataset_name, data) in enumerate(datasets.items()):
    ax = axes.flatten()[idx]
    y = np.arange(len(data["Methods"]))  # y 轴方法索引

    # ✅ 计算 Y 轴范围
    min_scc, max_scc = min(data["ALL_Scc"]), max(data["ALL_Scc"])
    min_wp, max_wp = min(data["WP_Nums"]), max(data["WP_Nums"])

    scc_margin = (max_scc - min_scc) * 0.2
    wp_margin = (max_wp - min_wp) * 0.2

    # ✅ 创建双 X 轴
    ax2 = ax.twiny()

    # ✅ ALL_Scc 柱形图（左 X 轴）
    ax.barh(y - bar_width / 2, data["ALL_Scc"], bar_width, color=colors["ALL_Scc"], alpha=0.9, label="ALL_Scc")

    # ✅ WP_Nums 柱形图（右 X 轴）
    ax2.barh(y + bar_width / 2, data["WP_Nums"], bar_width, color=colors["WP_Nums"], hatch=hatch_style, alpha=0.7, label="WP_Nums")

    # 📌 轴标签 & 添加子图标题 (a), (b), etc.
    ax.set_yticks(y)
    ax.set_yticklabels(data["Methods"], fontsize=tick_fontsize)
    ax.set_title(f"({chr(97 + idx)}) {dataset_name}", fontsize=label_fontsize + 2, pad=1)

    # ✅ 设定 X 轴范围（调整刻度）
    ax.set_xlim([round(min_scc - scc_margin, 2), round(max_scc + scc_margin, 2)])  # 统一保留两位小数
    ax2.set_xlim([min_wp - wp_margin, max_wp + wp_margin])

    # ✅ 统一 X 轴 & Y 轴实线
    ax.spines["bottom"].set_linestyle("-")
    ax.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_linestyle("-")
    ax2.spines["top"].set_visible(True)  # ✅ 右 X 轴（WP_Nums）现在有实线

# ✅ 添加全局标题（如果需要）
# fig.suptitle("Feature Reduction Experiment Results", fontsize=18, fontweight="bold")

# ✅ 保存 & 显示
plt.savefig("../feature_reduction_experiment_final.jpg", format="jpg", dpi=600, bbox_inches="tight")
plt.savefig("../feature_reduction_experiment_final.eps", format="eps", dpi=600, bbox_inches="tight")
plt.show()
