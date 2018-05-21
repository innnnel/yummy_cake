# coding=utf-8
from numpy import *

'''通过方差的百分比来计算将数据降到多少维是比较合适的，
函数传入的参数是特征值和百分比percentage，返回需要降到的维度数num'''


def eig_val_pct(eig_vals, percentage):
    sort_array = sort(eig_vals)  # 使用numpy中的sort()对特征值按照从小到大排序
    sort_array = sort_array[-1::-1]  # 特征值从大到小排序
    array_sum = sum(sort_array)  # 数据全部的方差arraySum
    temp_sum = 0
    num = 0
    for i in sort_array:
        temp_sum += i
        num += 1
        if temp_sum >= array_sum * percentage:
            return num


'''pca函数有两个参数，其中dataMat是已经转换成矩阵matrix形式的数据集，列表示特征；
其中的percentage表示取前多少个特征需要达到的方差占比，默认为0.9'''


def pca(data_mat, percentage=0.9):
    mean_vals = mean(data_mat, axis=0)  # 对每一列求平均值，因为协方差的计算中需要减去均值
    mean_removed = data_mat - mean_vals
    cov_mat = cov(mean_removed, rowvar=0)  # cov()计算方差
    eig_vals, eig_vects = linalg.eig(mat(cov_mat))  # 利用numpy中寻找特征值和特征向量的模块linalg中的eig()方法
    k = eig_val_pct(eig_vals, percentage)  # 要达到方差的百分比percentage，需要前k个向量
    eig_val_ind = argsort(eig_vals)  # 对特征值eigVals从小到大排序
    eig_val_ind = eig_val_ind[:-(k + 1):-1]  # 从排好序的特征值，从后往前取k个，这样就实现了特征值的从大到小排列
    red_eig_vects = eig_vects[:, eig_val_ind]  # 返回排序后特征值对应的特征向量redEigVects（主成分）
    lowd_data_mat = mean_removed * red_eig_vects  # 将原始数据投影到主成分上得到新的低维数据lowDDataMat
    recon_mat = (lowd_data_mat * red_eig_vects.T) + mean_vals  # 得到重构数据reconMat
    return lowd_data_mat, recon_mat
