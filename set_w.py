import numpy as np
import pandas as pd


def calculate_label_quality(matrix1, task_idx, nearest_indices, weights, worker_idx):
    """
    计算第一个工人对于特定任务的标签质量。

    :param matrix1: 任务-工人标签矩阵（pandas.DataFrame）
    :param task_idx: 任务的编号
    :param nearest_indices: 第一个工人的标记过的任务里面距离task_idx最近的K个任务的编号
    :param weights: K个任务的权重，表示任务与目标任务的相似度
    :param worker_idx: 第一个工人的索引
    :return: 第一个工人对于目标任务标签的质量w_ir
    """

    # 获取工人对目标任务的标签
    label = matrix1.at[task_idx, worker_idx]

    # 计算 b1_ir (目标任务的标签一致性)
    # 计算其他工人的标签一致性
    b1_ir = 0
    count = 0
    for r_prime in matrix1.columns:  # 遍历所有工人
        if r_prime != worker_idx:  # 排除自己
            other_label = matrix1.at[task_idx, r_prime]
            if other_label != -1 and other_label == label:  # 只有在标签一致且标签不为-1时才考虑
                b1_ir += 1
            if other_label != -1:  # 只统计标记过的任务
                count += 1

    if count > 0:
        b1_ir /= count  # 归一化为 [0, 1] 之间的值
    else:
        b1_ir = 0

    # 计算 b2_ir (K个邻近任务的标签一致性)
    b2_ir = 0
    Z = 0  # 归一化常数
    for neighbor_idx, weight in zip(nearest_indices, weights):
        neighbor_label = matrix1.at[neighbor_idx, worker_idx]
        if neighbor_label != -1 and neighbor_label == label:
            b2_ir += weight
        Z += weight

    if Z > 0:
        b2_ir /= Z  # 归一化为 [0, 1] 之间的值
    else:
        b2_ir = 0

    # 计算偏差 b_ir
    b_ir = (b1_ir + b2_ir) / 2

    # 计算方差 v_ir (目标任务及其K个邻近任务的标签一致性)
    v_ir = 0
    for neighbor_idx, weight in zip(nearest_indices, weights):
        neighbor_label = matrix1.at[neighbor_idx, worker_idx]
        if neighbor_label != -1 and neighbor_label == label:
            v_ir += weight
        Z += weight  # 归一化常数

    if Z > 0:
        v_ir /= Z  # 归一化为 [0, 1] 之间的值
    else:
        v_ir = 0

    # 计算标签质量 w_ir
    w_ir = (b_ir + v_ir) / 2

    return w_ir



