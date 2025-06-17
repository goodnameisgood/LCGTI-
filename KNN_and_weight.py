import numpy as np
import pandas as pd

def calculate_normalized_distance(x_i, x_j, max_vals, min_vals):
    """
    计算两个数值型向量之间的标准化欧氏距离。
    """
    distance = 0.0
    for m in range(len(x_i)):
        max_min_diff = max_vals[m] - min_vals[m]
        if max_min_diff != 0:  # 防止除以零
            distance += ((x_i[m] - x_j[m]) / max_min_diff) ** 2
    return np.sqrt(distance)

def calculate_nearest_tasks_and_weights(file_path, task_idx, K, task_indices_set):
    """
    计算给定任务 task_idx 的 K 个最近邻任务和权重，从指定的任务编号集合中选择。
    """
    # 读取数据
    data = pd.read_csv(file_path)

    # 转换数据为 numpy 数组
    dataset = data.values

    # 计算每个数值属性的最大值和最小值
    max_vals = np.max(dataset, axis=0)
    min_vals = np.min(dataset, axis=0)

    # 获取目标任务的特征向量
    task_vector = dataset[task_idx]

    # 计算与指定任务集合中任务的距离
    distances = []
    for j in task_indices_set:
        if j != task_idx:  # 排除目标任务自身
            distance = calculate_normalized_distance(task_vector, dataset[j], max_vals, min_vals)
            distances.append((j, distance))  # 存储任务编号和对应的距离

    # 根据距离进行排序，选择最近的K个任务
    distances.sort(key=lambda x: x[1])  # 按距离升序排序
    nearest_indices = [d[0] for d in distances[:K]]  # 取出最接近的K个任务编号
    nearest_distances = [d[1] for d in distances[:K]]  # 取出最接近的K个任务的距离

    # 计算最近 K 个任务的权重
    weights = 1 - np.array(nearest_distances) / nearest_distances[-1]  # 权重：距离越小，权重越大

    return nearest_indices, weights.tolist()
