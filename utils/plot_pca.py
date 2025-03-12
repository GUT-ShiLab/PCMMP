import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean

# 设置专业的绘图风格和字体
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

def plot_confidence_ellipse(x, y, ax, n_std=1.96, edgecolor="black", linestyle="--", linewidth=2):
    """绘制95%置信椭圆"""
    if x.size != y.size:
        raise ValueError("x and y must have the same size")
    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse((mean_x, mean_y), width, height, angle=angle,
                      edgecolor=edgecolor, facecolor="none",
                      linestyle=linestyle, linewidth=linewidth)
    ax.add_patch(ellipse)

def permutation_test(train_pca, test_pca, n_permutations=999):
    """置换检验计算分布差异的显著性p值"""
    combined = np.vstack([train_pca, test_pca])
    labels = np.array([0] * len(train_pca) + [1] * len(test_pca))
    obs_distance = euclidean(np.mean(train_pca, axis=0), np.mean(test_pca, axis=0))
    perm_distances = []
    for _ in range(n_permutations):
        np.random.shuffle(labels)
        perm_train = combined[labels == 0]
        perm_test = combined[labels == 1]
        perm_distance = euclidean(np.mean(perm_train, axis=0), np.mean(perm_test, axis=0))
        perm_distances.append(perm_distance)
    perm_distances = np.array(perm_distances)
    p_value = (np.sum(perm_distances >= obs_distance) + 1) / (n_permutations + 1)
    return p_value

def plot_pca_subplot(ax, train_data, test_data, title):
    """
    对训练集与测试集数据进行PCA降维，绘制散点图和95%置信椭圆，
    并计算两队列中心点的欧氏距离与置换检验p值。
    """
    # 合并数据（按样本维度）并填充缺失值为0
    combined = pd.concat([train_data, test_data], axis=0).fillna(0)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined.values)
    n_train = train_data.shape[0]
    train_pca = pca_result[:n_train]
    test_pca = pca_result[n_train:]

    # 计算中心点与欧氏距离
    centroid_train = np.mean(train_pca, axis=0)
    centroid_test = np.mean(test_pca, axis=0)
    distance = euclidean(centroid_train, centroid_test)

    # 置换检验
    p_value = permutation_test(train_pca, test_pca)

    # 绘制散点图：训练集用圆点，测试集用菱形
    ax.scatter(train_pca[:, 0], train_pca[:, 1], c="tab:blue", marker="o",
               edgecolor="k", s=70, label="Train", alpha=0.8)
    ax.scatter(test_pca[:, 0], test_pca[:, 1], c="tab:orange", marker="D",
               edgecolor="k", s=70, label="Test", alpha=0.8)

    # 绘制95%置信椭圆
    plot_confidence_ellipse(train_pca[:, 0], train_pca[:, 1], ax, n_std=1.96,
                            edgecolor="tab:blue", linestyle="--", linewidth=2)
    plot_confidence_ellipse(test_pca[:, 0], test_pca[:, 1], ax, n_std=1.96,
                            edgecolor="tab:orange", linestyle="--", linewidth=2)

    # 绘制中心点
    ax.scatter(centroid_train[0], centroid_train[1], c="tab:blue", marker="X", s=100,
               label="Train Centroid", edgecolor="k")
    ax.scatter(centroid_test[0], centroid_test[1], c="tab:orange", marker="X", s=100,
               label="Test Centroid", edgecolor="k")

    # 设置轴标签和标题
    ax.set_xlabel("PC1 (%.1f%%)" % (pca.explained_variance_ratio_[0] * 100))
    ax.set_ylabel("PC2 (%.1f%%)" % (pca.explained_variance_ratio_[1] * 100))
    ax.set_title(title, fontsize=16)
    ax.legend(frameon=True, framealpha=0.9)

    # 在图上标注距离和p值
    ax.text(0.05, 0.95, f"Euclidean Dist: {distance:.2f}\np: {p_value:.3f}",
            transform=ax.transAxes, fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    return distance, p_value

# ================== 主程序 ==================
if __name__ == "__main__":
    # 读取数据并转置（行=样本，列=特征）
    micro_train = pd.read_csv("../data/IBD/DL_data/microbiome_Train_CLR.csv", index_col=0).T
    micro_test  = pd.read_csv("../data/IBD/DL_data/microbiome_Test_CLR.csv", index_col=0).T
    metab_train = pd.read_csv("../data/IBD/CLR_unfiltered_data/metab_train_clr.csv", index_col=0).T
    metab_test  = pd.read_csv("../data/IBD/CLR_unfiltered_data/metab_test_clr.csv", index_col=0).T

    # 检查数据形状
    print("Microbiome Train shape:", micro_train.shape)
    print("Microbiome Test shape:", micro_test.shape)
    print("Metabolomics Train shape:", metab_train.shape)
    print("Metabolomics Test shape:", metab_test.shape)

    # 创建包含两个子图的画布
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))

    dist_micro, p_micro = plot_pca_subplot(axs[0], micro_train, micro_test, "(a) PCA of Microbiome Data")
    dist_metab, p_metab = plot_pca_subplot(axs[1], metab_train, metab_test, "(b) PCA of Metabolome Data")

    plt.tight_layout()
    # plt.savefig("cross_cohort_pca_analysis_improved.jpg", dpi=600, bbox_inches="tight")
    plt.savefig("cross_cohort_pca_analysis_improved.svg", dpi=600, bbox_inches="tight")
    plt.savefig("cross_cohort_pca_analysis_improved.tiff", dpi=600, bbox_inches="tight",pil_kwargs={"compression": "tiff_lzw"})
    plt.show()

    print("[Microbiome] Center distance: {:.2f}, p = {:.3f}".format(dist_micro, p_micro))
    print("[Metabolomics] Center distance: {:.2f}, p = {:.3f}".format(dist_metab, p_metab))
