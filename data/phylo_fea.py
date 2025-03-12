import pandas as pd
import numpy as np
from ete3 import NCBITaxa

# Step 1: 读取微生物丰度表
file_path = "./IBD/Origin_data/microbiome_Train.csv"
data = pd.read_csv(file_path)
# 提取特征名列和丰度列
feature_column = data.columns[0]
data[feature_column] = data[feature_column].astype(str).apply(lambda x: x.strip())

# 提取属名并合并丰度的函数
def extract_genus(feature_name):
    # 分割得到属名部分，并去掉可能存在的后缀
    genus_part = feature_name.split("g__")[-1]
    genus = genus_part.split("_")[0] if "_" in genus_part else genus_part
    return genus.strip()

# 创建一个字典来存储属名及其对应的丰度
genus_dict = {}

# 遍历所有特征名称并提取属名
for index, row in data.iterrows():
    genus = extract_genus(row[feature_column])
    abundance = row[1:].values.astype(float)  # 确保丰度值为浮点数类型

    if genus in genus_dict:
        genus_dict[genus] += abundance  # 合并相同属的丰度
    else:
        genus_dict[genus] = abundance  # 新增属名及对应丰度

# 将属名和丰度合并为 DataFrame，注意这里使用原始数据的列名作为新的列名
sample_names = data.columns[1:]  # 获取样本名作为列名
genus_names = list(genus_dict.keys())
abundance_values = np.vstack(list(genus_dict.values())).T # 转置以匹配样本名和属名的位置
genus_data = pd.DataFrame(data=abundance_values, columns=genus_names, index=sample_names)

# # 保存合并后的数据，设置index=True以保存样本名到csv的第一列
# output_path = "./data/IBD/phylogenTree_data/genus_merged.csv"
# genus_data.to_csv(output_path, index=True)
#
# print(f"合并后的属名丰度表已保存到: {output_path}")

# Step 2: 转换属名为 NCBI ID
raw_name = genus_data.columns.tolist()  # 属名
ncbi = NCBITaxa()

# 查找属名对应的 NCBI ID
raw_id = ncbi.get_name_translator(raw_name)

# 过滤掉没有 NCBI ID 的属名
valid_genus = []
valid_raw_id_list = []
for genus in raw_name:
    if genus in raw_id and len(raw_id[genus]) > 0:
        valid_genus.append(genus)
        valid_raw_id_list.append(str(raw_id[genus][0]))  # 获取第一个有效的 NCBI ID

# 删除没有对应 NCBI ID 的属名及其数据
genus_data = genus_data[valid_genus]
# **Step 2.1: 合并相同 NCBI ID 的特征**
merged_ncbi_dict = {}
for genus, ncbi_id in zip(valid_genus, valid_raw_id_list):
    abundance = genus_data[genus].values
    if ncbi_id in merged_ncbi_dict:
        merged_ncbi_dict[ncbi_id] += abundance
    else:
        merged_ncbi_dict[ncbi_id] = abundance

# 重新创建 DataFrame，确保 NCBI ID 作为唯一列名
genus_data = pd.DataFrame(data=np.vstack(list(merged_ncbi_dict.values())).T,
                          columns=list(merged_ncbi_dict.keys()),
                          index=genus_data.index)

# Step 3: 使用 NCBI 获取系统发育树并排序属名
tree = ncbi.get_topology(valid_raw_id_list)
tree_ascii = tree.get_ascii(attributes=["taxid"])
output_path_tree = "./IBD/phylogenTree_data/phylogenetic_tree.txt"
with open(output_path_tree, "w") as file:
    file.write(tree_ascii)

# 获取顺序
postorder = []
for node in tree.traverse(strategy='postorder'):
    if node.is_leaf():
        postorder.append(node.name)

# 根据顺序重新排序数据
ordered_genus_data_p = genus_data[postorder]
# 按行归一化函数
def normalize_to_one(df):
    numeric_df = df.copy()
    numeric_df = numeric_df.apply(pd.to_numeric, errors='coerce')  # 确保所有数据为数值类型
    row_sums = numeric_df.sum(axis=1)  # 计算每行的总和
    normalized_df = numeric_df.div(row_sums, axis=0)  # 按行归一化
    normalized_df.fillna(0, inplace=True)  # 填充可能的 NaN 值为 0
    return normalized_df

# 对排序后的数据进行归一化
normalized_genus_data_p = normalize_to_one(ordered_genus_data_p)

# 保存归一化后的数据
output_path_p = './IBD/phylogenTree_data/phylogenTree_p_Train.csv'

normalized_genus_data_p.to_csv(output_path_p)

print(f"系统发育树已保存为 {output_path_tree}")
print(f"归一化后的数据已保存为 {output_path_p}")