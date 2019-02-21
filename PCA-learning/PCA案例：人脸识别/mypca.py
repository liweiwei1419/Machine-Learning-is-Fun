import numpy as np


class PCA1:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_vals = None
        self.mean_removed = None
        # 主成分，即协方差矩阵特征值从大到小排序，按列排成的矩阵
        self.components_ = None
        # 能解释的方差，其实就是协方差矩阵特征值从大到小排序，组成的数组
        self.explained_variance_ = None

        self.low_d_data_mat = None

    def fit(self, X):
        mean_vals = np.mean(X, axis=0)
        mean_removed = X - mean_vals

        self.mean_vals = mean_vals
        self.mean_removed = mean_removed

        cov_mat = mean_removed.T.dot(
            mean_removed) / (mean_removed.shape[0] - 1)
        eig_vals, eig_vects = np.linalg.eig(cov_mat)
        eig_val_ind = np.argsort(eig_vals)
        eig_val_ind_desc = eig_val_ind[:-(self.n_components + 1):-1]
        # 【注意】eig_vects 这个矩阵的每一列是特征向量
        red_eig_vects = eig_vects[:, eig_val_ind_desc]
        self.components_ = red_eig_vects
        self.explained_variance_ = eig_vals[eig_val_ind_desc]

    def transform(self):
        self.low_d_data_mat = self.mean_removed.dot(self.components_)
        return self.low_d_data_mat + self.mean_vals

    def inverse_transform(self):
        return self.low_d_data_mat.dot(self.components_.T) + self.mean_vals


class PCA2:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean_vals = None
        self.mean_removed = None
        # 主成分，即协方差矩阵特征值从大到小排序，按列排成的矩阵
        self.components_ = None
        self.low_d_data_mat = None

    def fit(self, X):
        mean_vals = np.mean(X, axis=0)
        mean_removed = X - mean_vals
        self.mean_vals = mean_vals
        self.mean_removed = mean_removed
        U, S, VT = np.linalg.svd(mean_removed)
        self.components_ = VT.T[:, :self.components_]

    def transform(self):
        self.low_d_data_mat = self.mean_removed.dot(self.components_)
        return self.low_d_data_mat + self.mean_vals

    def inverse_transform(self):
        return self.low_d_data_mat.dot(self.components_.T) + self.mean_vals
