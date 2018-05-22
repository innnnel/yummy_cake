# from PCA import pca
import numpy as np


if __name__ == "__main__":
    # matrix = np.loadtxt("Ma.txt")
    # lowd_matrix, recon_matrix, red_eig_vects = pca(matrix)
    # np.savetxt("lowdmatrix.txt", lowd_matrix)
    # np.savetxt("reconmatrix.txt", recon_matrix)
    # np.savetxt("redeigvects.txt", red_eig_vects)

    turn_to_low = np.loadtxt("redeigvects.txt")
    matrix = np.loadtxt("新建文本文档.txt")
    mean_vals = np.mean(matrix, axis=0)  # 对每一列求平均值，因为协方差的计算中需要减去均值
    matrix = matrix - mean_vals
    matrix = np.dot(matrix, turn_to_low)
    np.savetxt("lowdmatrix2.txt", matrix)




