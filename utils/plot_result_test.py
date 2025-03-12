import matplotlib.pyplot as plt
import numpy as np

# 数据
methods = ["MelonnPan", "ENVIM", "MMINP", "MiMeNet", "BiomeNED", "mNODE", "LOCATE", "PCMMP"]
all_scc = [0.3501, 0.2731, 0.384, 0.234, 0.1497, 0.2487, 0.2213, 0.2906]
wp_nums = [592, 412, 836, 3218, 1636, 3669, 3110, 4263]

# 颜色设置
bar_color = "#1f77b4"  # 蓝色（柱状图）
line_color = "#ff7f0e"  # 橙色（折线图）
text_color="#000000"

fig, ax1 = plt.subplots(figsize=(10, 6))

# 柱状图（ALL_Scc）
bars = ax1.bar(methods, all_scc, color=bar_color, alpha=0.7, label="ALL_Scc", width=0.6)
ax1.set_ylabel("ALL_Scc", fontsize=14, color=text_color)
ax1.set_ylim(0, max(all_scc) * 1.3)
ax1.tick_params(axis='y', labelcolor=text_color)

# 在柱形图上方标注 ALL_Scc 值，并针对 BiomeNED 调整位置
for i, (bar, value) in enumerate(zip(bars, all_scc)):
    offset = -0.02 if methods[i] == "BiomeNED" else 0.01  # BiomeNED 上移避免重叠
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset, f"{value:.3f}",
             ha='center', va='bottom', fontsize=12, color=text_color)

# 次坐标轴（折线图 WP_Nums）
ax2 = ax1.twinx()
ax2.plot(methods, wp_nums, color=line_color, marker="o", linestyle="-", linewidth=2, markersize=6, label="WP_Nums")
ax2.set_ylabel("WP_Nums", fontsize=14, color=text_color)
ax2.set_ylim(0, max(wp_nums) * 1.2)
ax2.tick_params(axis='y', labelcolor=text_color)

# 在折线图节点处标注 WP_Nums 值，并调整 MiMeNet 和 BiomeNED 避免重叠
for i, value in enumerate(wp_nums):
    y_offset = 350 if methods[i] in ["BiomeNED"] else 150
    x_offset = -0.04 if methods[i] in ["BiomeNED","MMINP","LOCATE"] else 0  # BiomeNED 右偏一点，避免重叠
    ax2.text(i + x_offset, value + y_offset, f"{value}", ha='center', va='bottom', fontsize=12, color=text_color)

# 网格
ax1.grid(axis='y', linestyle="--", alpha=0.5)

# 标题和标签
# plt.title("Comparison of ALL_Scc and WP_Nums Across Methods", fontsize=14, fontweight='bold')
ax1.set_xlabel("Methods", fontsize=14)  # 添加 X 轴标题

# 图例
ax1.legend(loc="upper left", fontsize=12)
ax2.legend(loc="upper right", fontsize=12)

# 调整布局
plt.tight_layout()

# 保存图表（SVG 格式）
output_path = "../method_comparison.svg"
# plt.savefig(output_path, format="jpg", dpi=600)
plt.savefig(output_path, format="svg", dpi=600)
print(f"✅ 图表已保存至 {output_path}")

# 显示图表
plt.show()
