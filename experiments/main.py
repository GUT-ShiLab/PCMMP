import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from models.regressor_MLP import SimpleConvMLPRegressor,MLPRegressor
from models.combined_model import train_epoch, evaluate
from losses.losses import MSELossNoContrastive
from math import sqrt
import os

# ========================== 数据集加载 ==========================
metadata_path = "../data/IBD/CLR_unfiltered_data/metadata_IBD.csv"
microbiome_path = "../data/IBD/DL_data/microbiome_Train_CLR.csv"
metabolome_path = "../data/IBD/CLR_unfiltered_data/metab_train_clr.csv"
external_microbiome_path = "../data/IBD/DL_data/microbiome_Test_CLR.csv"
external_metabolome_path = "../data/IBD/CLR_unfiltered_data/metab_test_clr.csv"
# ==========================  确保数据存在 ==========================
for path in [metadata_path, microbiome_path, metabolome_path]:
    assert os.path.exists(path), f"❌ 错误：文件 {path} 不存在！"

if external_microbiome_path:
    assert os.path.exists(external_microbiome_path), f"❌ 错误：外部验证微生物数据 {external_microbiome_path} 不存在！"
if external_metabolome_path:
    assert os.path.exists(external_metabolome_path), f"❌ 错误：外部验证代谢数据 {external_metabolome_path} 不存在！"

# ========================== 加载参数 ==========================
batch_size = 32
num_epochs = 1000
early_stop_patience = 20  # 早停策略：10个epoch内无改善则停止
learning_rate = 1e-3  # 学习率
weight_decay = 1e-4  # L2正则化（权重衰减）
random_seed = 123


# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据集
dataset = MultiOmicsDataset(microbiome_path, metabolome_path, metadata_path=None,
                            external_microbiome_path=external_microbiome_path, external_metabolome_path=external_metabolome_path)

# 获取微生物和代谢物特征数量
microbiome_data = pd.read_csv(microbiome_path, index_col=0)
metabolome_data = pd.read_csv(metabolome_path, index_col=0)
microbiome_input_dim = microbiome_data.shape[0]  # 微生物特征数量
metabolome_output_dim = metabolome_data.shape[0]  # 代谢物输出特征数量
print("微生物特征数量：", microbiome_input_dim)
print("代谢物特征数量：", metabolome_output_dim)

# ========================== 交叉验证 ==========================

# 使用KFold进行交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

fold_results = {f"Fold_{i}_Scc": [] for i in range(5)}

# 逐个添加 RMSE, MAAPE, NMSE 结果存储
for i in range(5):
    fold_results[f"Fold_{i}_RMSE"] = []
    fold_results[f"Fold_{i}_MAAPE"] = []
    fold_results[f"Fold_{i}_NMSE"] = []
metabolite_names = None

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    print(f"Training fold {fold + 1}...")
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_idx)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_idx)
    # model = MLPRegressor(input_dim=microbiome_input_dim, hidden_dim=256, output_dim=metabolome_output_dim).to(device)
    model = SimpleConvMLPRegressor(input_dim=microbiome_input_dim, conv_dims=[32,64,128], mlp_dims=[256,128],
                                   output_dim=metabolome_output_dim, kernel_size=5, dropout_rate=0.4).to(device)
    criterion = MSELossNoContrastive().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Fold {fold}, Epoch {epoch}, Loss: {train_loss}")
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at fold {fold}, epoch {epoch}")
                break

    metabolite_names, fold_scc, _, _, fold_rmse, fold_maape, fold_nmse = evaluate(model, test_loader, device)
    fold_results[f"Fold_{fold}_Scc"] = fold_scc
    fold_results[f"Fold_{fold}_RMSE"] = fold_rmse
    fold_results[f"Fold_{fold}_MAAPE"] = fold_maape
    fold_results[f"Fold_{fold}_NMSE"] = fold_nmse

result_df = pd.DataFrame({"Metabolites": metabolite_names})
# 添加各指标的平均值列
for metric in ["Scc", "RMSE", "MAAPE", "NMSE"]:
    result_df[f"Average_{metric}"] = np.mean([fold_results[f"Fold_{i}_{metric}"] for i in range(5)], axis=0)

# 按平均Spearman相关性排序
result_df = result_df.sort_values(by="Average_Scc", ascending=False)
five_fold_metrics_results=f"../results/five_fold_metrics_results.csv"
# 导出结果到CSV文件
result_df[["Metabolites", "Average_Scc", "Average_RMSE", "Average_MAAPE", "Average_NMSE"]].to_csv(five_fold_metrics_results, index=False)

# 计算各指标的均值
avg_spearman_corr_all = np.mean(result_df["Average_Scc"])
avg_rmse_all = np.mean(result_df["Average_RMSE"])
avg_maape_all = np.mean(result_df["Average_MAAPE"])
avg_nmse_all = np.mean(result_df["Average_NMSE"])
# 计算 Spearman 相关性排名前 50 的均值
top_50_spearman = result_df["Average_Scc"].head(50)
avg_spearman_corr_top50 = np.mean(top_50_spearman)

# 计算 Spearman 相关性 ≥ 0.3 的代谢物数量
num_metabolites_ge_0_3 = (result_df["Average_Scc"] >= 0.3).sum()
proportion_ge_0_3 = num_metabolites_ge_0_3 / metabolome_output_dim

print("=========================================五折交叉==============================================")
print(f"Five-Fold Average Spearman Correlation (All Metabolites): {avg_spearman_corr_all:.4f}")
print(f"Five-Fold Average Spearman Correlation (Top 50 Metabolites): {avg_spearman_corr_top50:.4f}")
print(f"Number of Metabolites with Spearman Correlation >= 0.3: {num_metabolites_ge_0_3}")
print(f"Proportion of Metabolites with Average Spearman Correlation >= 0.3: {proportion_ge_0_3:.2%}")
# print(f"Five-Fold Average RMSE (All Metabolites): {avg_rmse_all:.4f} ± {std_rmse_all:.4f}")
print(f"Five-Fold Average MAAPE (All Metabolites): {avg_maape_all:.4f}")
# print(f"Five-Fold Average NMSE (All Metabolites): {avg_nmse_all:.4f} ± {std_nmse_all:.4f}")

# ========================== 外部验证 ==========================

# 外部验证集评估
external_microbiome, external_metabolome = dataset.get_external_validation_data()
external_microbiome = torch.tensor(external_microbiome.values, dtype=torch.float32).to(device)
external_metabolome = torch.tensor(external_metabolome.values, dtype=torch.float32).to(device)
print(f"External microbiome shape: {external_microbiome.shape}")

# 重新训练最终模型
# final_model = MLPRegressor(input_dim=microbiome_input_dim, hidden_dim=256, output_dim=metabolome_output_dim).to(device)
final_model = SimpleConvMLPRegressor(input_dim=microbiome_input_dim, conv_dims= [32,64,128], mlp_dims= [256,128],
                                     output_dim=metabolome_output_dim, kernel_size= 5, dropout_rate= 0.4).to(device)

final_criterion = MSELossNoContrastive().to(device)
final_optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

final_train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

best_loss = float('inf')
patience_counter = 0
for epoch in range(num_epochs):
    final_train_loss = train_epoch(final_model, final_train_loader, final_optimizer, final_criterion, device)
    if final_train_loss < best_loss:
        best_loss = final_train_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at final model, epoch {epoch}")
            break

# 外部验证集预测
with torch.no_grad():
    final_model.eval()
    external_predictions = final_model(external_microbiome).cpu().numpy()
    external_targets = external_metabolome.cpu().numpy()
    spearman_corr_external = [spearmanr(external_predictions[:, i], external_targets[:, i])[0] for i in range(external_targets.shape[1])]
    external_rmse = [sqrt(np.mean((external_predictions[:, i] - external_targets[:, i]) ** 2)) for i in range(external_targets.shape[1])]
    external_maape = [np.mean(np.arctan(np.abs((external_targets[:, i] - external_predictions[:, i]) / external_targets[:, i]))) * 2 / np.pi for i in range(external_targets.shape[1])]
    external_nmse = [np.mean((external_targets[:, i] - external_predictions[:, i]) ** 2) / np.var(external_targets[:, i]) if np.var(external_targets[:, i]) > 0 else np.nan for i in range(external_targets.shape[1])]
    external_results_df = pd.DataFrame({"Metabolites": metabolite_names, "Spearman_Correlation": spearman_corr_external,
                                        "RMSE": external_rmse, "MAAPE": external_maape, "NMSE": external_nmse})
    # external_validation_metrics_result=f"../results/external_validation_metrics_results.csv"
    external_metabolome_predictions=f"../results/external_metabolome_predictions.csv"
    # external_results_df.to_csv(external_validation_metrics_result, index=False)
    pd.DataFrame(external_predictions, columns=metabolite_names).to_csv(external_metabolome_predictions, index=False)

    num_metabolites_ge_0_3 = sum(np.array(spearman_corr_external) >= 0.3)
    proportion_ge_0_3 = num_metabolites_ge_0_3 / len(spearman_corr_external)
    top_50_avg_spearman = external_results_df.nlargest(50, "Spearman_Correlation")["Spearman_Correlation"].mean()
    print("=========================================测试集预测==============================================")
    # 计算并打印结果
    avg_spearman_corr_external_all = np.mean(spearman_corr_external)
    avg_rmse_external_all = np.mean(external_rmse)
    avg_maape_external_all = np.mean(external_maape)
    avg_nmse_external_all = np.mean(external_nmse)
    print(f"External Validation Average Spearman Correlation (All Metabolites): {avg_spearman_corr_external_all}")
    print(f"Proportion of Metabolites with Spearman Correlation >= 0.3 (External): {proportion_ge_0_3:.2%}")
    print(f"Numbers of Metabolites with Spearman Correlation >= 0.3 (External): {num_metabolites_ge_0_3}")
    # print(f"External Validation Average RMSE (All Metabolites): {avg_rmse_external_all}")
    print(f"External Validation Average MAAPE (All Metabolites): {avg_maape_external_all}")
    # print(f"External Validation Average NMSE (All Metabolites): {avg_nmse_external_all}")
    print(f"External Validation Average Spearman Correlation (Top 50 Metabolites): {top_50_avg_spearman}")

