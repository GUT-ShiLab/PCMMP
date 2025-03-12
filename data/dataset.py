import pandas as pd
import torch
from torch.utils.data import Dataset

class MultiOmicsDataset(Dataset):
    """Custom dataset for microbiome and metabolome data without labels."""
    def __init__(self, microbiome_path, metabolome_path, metadata_path=None,
                 external_microbiome_path=None, external_metabolome_path=None):
        """
        Args:
            microbiome_path (str): Path to microbiome data (CSV).
            metabolome_path (str): Path to metabolome data (CSV).
            metadata_path (str, optional): Path to metadata (CSV), not used in this version.
            external_microbiome_path (str, optional): Path to external microbiome test set (CSV).
            external_metabolome_path (str, optional): Path to external metabolome test set (CSV).
        """
        # 加载训练集数据
        self.microbiome_data = pd.read_csv(microbiome_path, index_col=0)
        self.metabolome_data = pd.read_csv(metabolome_path, index_col=0)

        # 对齐训练集的样本
        self.samples = self.microbiome_data.columns
        self.microbiome_data = self.microbiome_data.T
        self.metabolome_data = self.metabolome_data.T

        # 加载外部验证集数据（如果有）
        self.external_microbiome_data = None
        self.external_metabolome_data = None
        if external_microbiome_path and external_metabolome_path:
            external_microbiome_data = pd.read_csv(external_microbiome_path, index_col=0)
            external_metabolome_data = pd.read_csv(external_metabolome_path, index_col=0)

            print(f"External microbiome shape before: {external_microbiome_data.shape}")
            # train_features = set(self.microbiome_data.columns)
            # 对外部验证集筛选特征，保留训练集中存在的特征
            self.external_microbiome_data = external_microbiome_data.loc[self.microbiome_data.columns].T
            self.external_metabolome_data = external_metabolome_data.loc[self.metabolome_data.columns].T
            # external_features = set(self.external_microbiome_data.columns)
            print(f"External microbiome shape after: {self.external_microbiome_data.shape}")
            # missing_features = train_features - external_features
            # extra_features = external_features - train_features
            # print(f"Missing features in external: {missing_features}")
            # print(f"Extra features in external: {extra_features}")
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        microbiome = torch.tensor(self.microbiome_data.loc[sample].values, dtype=torch.float32)
        metabolome = torch.tensor(self.metabolome_data.loc[sample].values, dtype=torch.float32)
        # No label to return in this version
        return microbiome, metabolome

    def get_external_validation_data(self):
        """Return external validation set."""
        if self.external_microbiome_data is not None and self.external_metabolome_data is not None:
            return self.external_microbiome_data, self.external_metabolome_data
        else:
            raise ValueError("External validation data is not provided.")
