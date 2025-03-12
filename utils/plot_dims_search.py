import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# ------------------ 示例数据：IBD ------------------
data_ibd = [
    ("[128, 64]", "[32, 64]", 0.2917),
    ("[128, 64]", "[16, 32, 64]", 0.3422),
    ("[128, 64]", "[64, 128]", 0.3055),
    ("[128, 64]", "[32, 64, 128]", 0.3355),
    ("[128, 64]", "[128, 256]", 0.3037),
    ("[128, 64]", "[64, 128, 256]", 0.3170),
    ("[128, 64]", "[16, 32, 64, 128]", 0.3219),
    ("[128, 64]", "[32, 64, 128, 256]", 0.3462),

    ("[256, 128]", "[32, 64]", 0.3537),
    ("[256, 128]", "[16, 32, 64]", 0.3708),
    ("[256, 128]", "[64, 128]", 0.3505),
    ("[256, 128]", "[32, 64, 128]", 0.3739),
    ("[256, 128]", "[128, 256]", 0.3393),
    ("[256, 128]", "[64, 128, 256]", 0.3638),
    ("[256, 128]", "[16, 32, 64, 128]", 0.3539),
    ("[256, 128]", "[32, 64, 128, 256]", 0.3499),

    ("[512, 256]", "[32, 64]", 0.3581),
    ("[512, 256]", "[16, 32, 64]", 0.3364),
    ("[512, 256]", "[64, 128]", 0.3423),
    ("[512, 256]", "[32, 64, 128]", 0.3321),
    ("[512, 256]", "[128, 256]", 0.3517),
    ("[512, 256]", "[64, 128, 256]", 0.3318),
    ("[512, 256]", "[16, 32, 64, 128]", 0.3246),
    ("[512, 256]", "[32, 64, 128, 256]", 0.3135),
]

# ------------------ 示例数据：ESRD ------------------
data_esrd = [
    ("[128, 64]", "[32, 64]", 0.2230),
    ("[128, 64]", "[16, 32, 64]", 0.2282),
    ("[128, 64]", "[64, 128]", 0.2044),
    ("[128, 64]", "[32, 64, 128]", 0.2306),
    ("[128, 64]", "[128, 256]", 0.1835),
    ("[128, 64]", "[64, 128, 256]", 0.2222),
    ("[128, 64]", "[16, 32, 64, 128]", 0.1841),
    ("[128, 64]", "[32, 64, 128, 256]", 0.2096),

    ("[256, 128]", "[32, 64]", 0.2294),
    ("[256, 128]", "[16, 32, 64]", 0.2415),
    ("[256, 128]", "[64, 128]", 0.2382),
    ("[256, 128]", "[32, 64, 128]", 0.2478),
    ("[256, 128]", "[128, 256]", 0.1974),
    ("[256, 128]", "[64, 128, 256]", 0.2244),
    ("[256, 128]", "[16, 32, 64, 128]", 0.2181),
    ("[256, 128]", "[32, 64, 128, 256]", 0.1888),

    ("[512, 256]", "[32, 64]", 0.2178),
    ("[512, 256]", "[16, 32, 64]", 0.2043),
    ("[512, 256]", "[64, 128]", 0.2114),
    ("[512, 256]", "[32, 64, 128]", 0.2130),
    ("[512, 256]", "[128, 256]", 0.2074),
    ("[512, 256]", "[64, 128, 256]", 0.2098),
    ("[512, 256]", "[16, 32, 64, 128]", 0.1962),
    ("[512, 256]", "[32, 64, 128, 256]", 0.1748),
]


def parse_conv_dims(conv_str):
    """
    将类似 '[16, 32, 64]' 的字符串解析为整数列表，并返回排序 key:
    key = (层数, 依次升序的通道值)
    """
    nums = re.findall(r'\d+', conv_str)  # 提取所有数字
    nums = list(map(int, nums))
    return (len(nums), nums)


def plot_param_heatmap(data, ax, title):
    """
    将 (mlp_dims, conv_dims, ALL_SCC) 数据转换为 DataFrame 并绘制分组柱状图，
    同时按照自定义顺序对 conv_dims 进行排序，并为三种不同的 MLP 配置分别赋予不同的填充图案。
    """
    df = pd.DataFrame(data, columns=["mlp_dims", "conv_dims", "ALL_SCC"])
    # 以 conv_dims 为行，mlp_dims 为列，ALL_SCC 为值
    pivot_df = df.pivot(index="conv_dims", columns="mlp_dims", values="ALL_SCC")
    # 根据自定义顺序对行（conv_dims）重新排序
    sorted_index = sorted(pivot_df.index, key=parse_conv_dims)
    pivot_df = pivot_df.reindex(sorted_index)

    # 获取列顺序（MLP 配置），预期有三种配置
    mlp_configs = pivot_df.columns.tolist()
    # 限定三个不同的填充图案
    hatch_patterns = {mlp_configs[0]: "/", mlp_configs[1]: "\\", mlp_configs[2]: "x"}

    # 绘制分组柱状图
    pivot_df.plot(kind='bar', ax=ax, colormap="Set2", edgecolor="black", width=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Conv_dims", fontsize=13)
    ax.set_ylabel("ALL_SCC", fontsize=13)
    # 设置图例为 2x2 格式，并增大文字
    ax.legend(bbox_to_anchor=(0.86, 1), loc="upper center", ncol=2,
              fontsize=12, title_fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')

    # 为每个柱子添加对应的填充图案
    for i, container in enumerate(ax.containers):
        # 根据柱状图的顺序选择对应的填充图案（仅对前三个 mlp 配置生效）
        mlp_config = mlp_configs[i] if i < len(mlp_configs) else None
        hatch = hatch_patterns.get(mlp_config, "")
        for patch in container.patches:
            patch.set_hatch(hatch)

    # # 在柱子上方标注数值
    # for container in ax.containers:
    #     ax.bar_label(container, fmt="%.3f", label_type='edge', fontsize=9, rotation=90)

    # 获取最大 ALL_SCC 值，并绘制水平虚线
    max_val = df["ALL_SCC"].max()
    ax.axhline(y=max_val, color='red', linestyle='--', linewidth=1.5, label=f"Max: {max_val:.3f}")
    ax.legend(bbox_to_anchor=(0.86, 1), loc="upper center", ncol=2,
              fontsize=12, title_fontsize=12)

    ax.set_ylim(0, max_val * 1.2)


# ----------------- 主程序示例 -----------------
if __name__ == "__main__":
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))

    # IBD 数据集
    plot_param_heatmap(data_ibd, axs[0], title="(a) IBD Dataset")
    # ESRD 数据集
    plot_param_heatmap(data_esrd, axs[1], title="(b) ESRD Dataset")

    plt.tight_layout()
    plt.savefig("conv_mlp_param_search.svg", dpi=600, format='svg')
    plt.savefig("conv_mlp_param_search.eps", dpi=600, format='eps')
    # plt.savefig("conv_mlp_param_search.emf", dpi=600, format='emf', bbox_inches='tight')
    plt.show()
