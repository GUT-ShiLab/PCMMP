import pandas as pd

# 读取原始CSV文件
file_path = "./IBD/phylogenTree_data/phylogenTree_Train_CLR.csv"  # 替换为您的文件路径
data = pd.read_csv(file_path, index_col=0)

# 如果需要转置（特征为行，样本为列），取消注释下面代码：
# data = data.T

# 计算特征数量以及每组应包含的特征数（分成5组）
num_features = len(data)
remainder = num_features % 5
padding_needed = 5 - remainder if remainder != 0 else 0

# 如果特征数量不能被5整除，则采用首尾交替填充策略
if padding_needed:
    # 保存首个和最后一个特征的拷贝
    first_feature = data.iloc[0]
    last_feature = data.iloc[-1]
    pad_rows = []
    for i in range(padding_needed):
        # 偶数位置复制第一个特征，奇数位置复制最后一个特征
        if i % 2 == 0:
            pad_rows.append(first_feature)
        else:
            pad_rows.append(last_feature)
    # 将填充数据转换为DataFrame并生成新的行索引
    pad_df = pd.DataFrame(pad_rows)
    pad_df.index = [f'pad_{i}' for i in range(padding_needed)]
    # 拼接填充数据到原始数据
    data = pd.concat([data, pad_df])

# 重新计算总特征数以及每组特征数
num_features = len(data)
group_size = num_features // 5

# 初始化新的特征顺序列表
new_order = []
# 依次遍历每个组内的特征位置，再按照组顺序交叉添加索引
for i in range(group_size):
    for group in range(5):
        new_order.append(group * group_size + i)

# 根据新顺序重新排列数据
reordered_data = data.iloc[new_order]

# 导出重新排序后的CSV文件
output_path = "./IBD/DL_data/microbiome_Train_group_CLR.csv"  # 替换为您的导出路径
reordered_data.to_csv(output_path)

print(f"Reordered data has been saved to {output_path}")
