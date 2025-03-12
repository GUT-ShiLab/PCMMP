import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def clr_transform(df):
    """
    Perform CLR transformation on a DataFrame where columns represent samples.

    Parameters:
    - df: DataFrame with rows as features and columns as samples.

    Returns:
    - clr_df: DataFrame with CLR-transformed data.
    """
    clr_df = df.copy()
    for col in clr_df.columns:
        sample = clr_df[col]
        min_non_zero = sample[sample > 0].min()
        if pd.isna(min_non_zero):
            clr_df[col] = 0
            continue
        sample[sample == 0] = min_non_zero / 2
        geom_mean = (sample.prod()) ** (1 / len(sample))
        if geom_mean == 0:
            geom_mean = min_non_zero / 2
        clr_df[col] = sample.apply(lambda x: np.log(x / geom_mean))
    return clr_df

# Load the data
file_path = "./IBD/phylogenTree_data/phylogenTree_p_Train.csv"
data = pd.read_csv(file_path, index_col=0)


# Perform CLR transformation on PCA-transformed data
clr_pca_data = clr_transform(data)

# Save the transformed data
output_path = "./IBD/phylogenTree_data/phylogenTree_Train_CLR.csv"
clr_pca_data.to_csv(output_path)
print(f"CLR-transformed PCA data has been saved to: {output_path}")


