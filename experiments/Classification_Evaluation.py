import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# ✅ 读取数据
predicted_metabolome_path = "../results/external_metabolome_predictions_PCMMP.csv"
test_original_microbiome_path = "../data/IBD/CLR_unfiltered_data/micro_test_clr.csv"
test_original_metabolome_path = "../data/IBD/CLR_unfiltered_data/metab_test_clr.csv"
predicted_MLP_metabolome_path = "../results/external_metabolome_predictions_MLP.csv"
test_metadata_path = "../data/IBD/CLR_unfiltered_data/external_metadata_IBD.csv"

# ✅ 加载数据
test_original_microbiome = pd.read_csv(test_original_microbiome_path, index_col=0).values.T
test_original_metabolome = pd.read_csv(test_original_metabolome_path, index_col=0).values.T
predicted_MLP_metabolome = pd.read_csv(predicted_MLP_metabolome_path, index_col=None).values
predicted_metabolome = pd.read_csv(predicted_metabolome_path, index_col=None).values

test_metadata = pd.read_csv(test_metadata_path, index_col=0)
test_labels = test_metadata["Study.Group"].map(lambda x: 1 if x in ["UC", "CD"] else 0).values  # 1=疾病, 0=健康

# ✅ 数据集字典
datasets = {
    "Measured Microbiome": test_original_microbiome,
    "Measured Metabolome": test_original_metabolome,
    "MLP Predicted Metabolome": predicted_MLP_metabolome,
    "PCMMP Predicted Metabolome": predicted_metabolome
}

# ✅ 交叉验证设置
kf = KFold(n_splits=5, shuffle=True, random_state=123)

# ✅ 结果存储
results_rf = []
results_lr = []

# ✅ 遍历每个数据集
for dataset_name, data in datasets.items():
    accuracies_rf, aucs_rf, precisions_rf, recalls_rf, f1s_rf = [], [], [], [], []
    accuracies_lr, aucs_lr, precisions_lr, recalls_lr, f1s_lr = [], [], [], [], []

    for train_index, test_index in kf.split(data):
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = test_labels[train_index], test_labels[test_index]

        # ✅ 训练 LASSO 逻辑回归（L1 正则化）
        clf_lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=123)
        clf_lr.fit(X_train, y_train)

        # ✅ 预测（LASSO 逻辑回归）
        y_pred_lr = clf_lr.predict(X_test)
        y_prob_lr = clf_lr.predict_proba(X_test)[:, 1]

        # ✅ 计算分类指标（LASSO 逻辑回归）
        accuracies_lr.append(accuracy_score(y_test, y_pred_lr))
        aucs_lr.append(roc_auc_score(y_test, y_prob_lr))
        precisions_lr.append(precision_score(y_test, y_pred_lr))
        recalls_lr.append(recall_score(y_test, y_pred_lr))
        f1s_lr.append(f1_score(y_test, y_pred_lr))


    # ✅ 计算五折交叉验证均值（LASSO 逻辑回归）
    mean_acc_lr = np.mean(accuracies_lr)
    mean_auc_lr = np.mean(aucs_lr)
    mean_precision_lr = np.mean(precisions_lr)
    mean_recall_lr = np.mean(recalls_lr)
    mean_f1_lr = np.mean(f1s_lr)

    results_lr.append([dataset_name, mean_acc_lr, mean_auc_lr, mean_precision_lr, mean_recall_lr, mean_f1_lr])
    print(f"📌 {dataset_name} (LASSO Logistic Regression): Accuracy = {mean_acc_lr:.4f}, AUC = {mean_auc_lr:.4f}, Precision = {mean_precision_lr:.4f}, Recall = {mean_recall_lr:.4f}, F1-score = {mean_f1_lr:.4f}\n")

# ✅ 结果整理 & 输出
df_results_lr = pd.DataFrame(results_lr, columns=["Dataset", "Mean Accuracy", "Mean AUC", "Mean Precision", "Mean Recall", "Mean F1-score"])
df_results_lr.to_csv("../results/lasso_logistic_classification_results.csv", index=False)

print("\n✅ 交叉验证完成，结果已保存到:")
print(" - '../results/lasso_logistic_classification_results.csv' (LASSO Logistic Regression)")
