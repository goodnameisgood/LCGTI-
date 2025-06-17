import torch
import pandas as pd
import numpy as np
import os
import time # 用于计时比较

# --- 设备选择 ---
from sklearn.metrics import f1_score

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("使用 GPU 进行计算")
else:
    device = torch.device("cpu")
    print("使用 CPU 进行计算")

# --- 优化后的距离计算函数 (GPU 加速) ---
def calculate_normalized_distance_gpu(x_i, x_j_batch, max_vals, min_vals):
    """
    计算一个数值型向量 x_i 与一批向量 x_j_batch 之间的标准化欧氏距离。
    使用 PyTorch 在 GPU 上进行计算。

    Args:
        x_i (torch.Tensor): 目标任务的特征向量 (1, num_features)。
        x_j_batch (torch.Tensor): 一批待比较任务的特征向量 (batch_size, num_features)。
        max_vals (torch.Tensor): 每个特征的最大值 (1, num_features)。
        min_vals (torch.Tensor): 每个特征的最小值 (1, num_features)。

    Returns:
        torch.Tensor: x_i 与 x_j_batch 中每个向量的距离 (batch_size,)。
    """
    # 确保所有张量在同一设备上
    x_i = x_i.to(device)
    x_j_batch = x_j_batch.to(device)
    max_vals = max_vals.to(device)
    min_vals = min_vals.to(device)

    # 计算 max_min_diff，避免除以零
    max_min_diff = max_vals - min_vals
    # 增加一个小的 epsilon 防止除以零，或者使用 torch.where
    eps = 1e-8
    max_min_diff = torch.where(max_min_diff == 0, eps, max_min_diff)
    # print(f"x_i shape: {x_i.shape}") # 应该是 (1, num_features)
    # print(f"x_j_batch shape: {x_j_batch.shape}") # 应该是 (N, num_features)
    # print(f"max_min_diff shape: {max_min_diff.shape}") # 应该是 (1, num_features)

    # 使用广播机制进行计算 (x_i - x_j_batch)
    # (N, num_features) - (1, num_features) -> (N, num_features)
    diff = x_i - x_j_batch
    # print(f"diff shape: {diff.shape}")

    # 归一化 (N, num_features) / (1, num_features) -> (N, num_features)
    normalized_diff = diff / max_min_diff
    # print(f"normalized_diff shape: {normalized_diff.shape}")

    # 平方 -> (N, num_features)
    squared_diff = normalized_diff ** 2
    # print(f"squared_diff shape: {squared_diff.shape}")

    # 沿特征维度求和 -> (N,)
    distance_sq = torch.sum(squared_diff, dim=1)
    # print(f"distance_sq shape: {distance_sq.shape}")

    # 开方 -> (N,)
    distance = torch.sqrt(distance_sq)
    # print(f"distance shape: {distance.shape}")

    return distance

# --- 优化后的 K 近邻和权重计算函数 (GPU 加速) ---
def calculate_nearest_tasks_and_weights_gpu(task_feature_tensor, task_idx, K, task_indices_set):
    """
    计算给定任务 task_idx 的 K 个最近邻任务和权重，从指定的任务编号集合中选择。
    使用 PyTorch 在 GPU 上进行计算。

    Args:
        task_feature_tensor (torch.Tensor): 所有任务的特征张量 (num_tasks, num_features)。
        task_idx (int): 目标任务的索引。
        K (int): 需要找到的最近邻数量。
        task_indices_set (np.ndarray or list): 候选任务的索引集合。

    Returns:
        tuple: (最近邻任务索引列表, 最近邻任务权重列表)
               如果候选任务数量不足 K，则返回所有候选任务。
    """
    # 确保 task_feature_tensor 在正确的设备上
    task_feature_tensor = task_feature_tensor.to(device)

    # 计算每个数值属性的最大值和最小值 (在 GPU 上计算)
    # Keepdim=True 保持维度为 (1, num_features) 以便广播
    max_vals = torch.max(task_feature_tensor, dim=0, keepdim=True)[0]
    min_vals = torch.min(task_feature_tensor, dim=0, keepdim=True)[0]

    # 获取目标任务的特征向量
    # unsqueeze(0) 增加一个维度，变为 (1, num_features)
    task_vector = task_feature_tensor[task_idx].unsqueeze(0).to(device)

    # 将候选任务索引转换为 LongTensor
    candidate_indices = torch.tensor(list(task_indices_set), dtype=torch.long, device=device)

    # 排除目标任务自身 (如果存在于候选集中)
    candidate_indices = candidate_indices[candidate_indices != task_idx]

    # 如果没有候选任务或候选任务太少
    if len(candidate_indices) == 0:
        return [], []
    if len(candidate_indices) < K:
        K = len(candidate_indices) # 调整 K 为实际候选数量

    # 获取候选任务的特征向量
    candidate_vectors = task_feature_tensor[candidate_indices].to(device)

    # 批量计算距离 (GPU 加速)
    distances_tensor = calculate_normalized_distance_gpu(task_vector, candidate_vectors, max_vals, min_vals)

    # 如果只有一个候选，特殊处理排序
    if distances_tensor.numel() == 1:
         nearest_indices_gpu = candidate_indices
         nearest_distances_gpu = distances_tensor
    else:
        # 根据距离进行排序，选择最近的K个任务 (在 GPU 上排序)
        # torch.sort 返回 (排序后的值, 排序后的索引)
        sorted_distances, sorted_indices_in_candidates = torch.sort(distances_tensor)

        # 选择前 K 个
        nearest_distances_gpu = sorted_distances[:K]
        # 获取原始任务索引中对应的索引
        nearest_original_indices = sorted_indices_in_candidates[:K]
        # 映射回原始任务编号
        nearest_indices_gpu = candidate_indices[nearest_original_indices]

    # 处理距离全为0的特殊情况 (所有邻居都完全相同)
    if torch.all(nearest_distances_gpu == 0):
         weights = torch.ones_like(nearest_distances_gpu) / K # 均匀分配权重
    elif nearest_distances_gpu[-1] == 0: # 防止除以0，如果最远的距离是0
        # 这种情况理论上不应该发生，除非K=1且距离为0，上面已处理
        # 或者多个最近邻距离都是0，最远的是0
        weights = torch.where(nearest_distances_gpu == 0, 1.0 / K, 0.0) # 给距离为0的分配权重
    else:
        # 计算最近 K 个任务的权重 (在 GPU 上计算)
        # 权重：1 - d / d_max (d_max 是第 K 个邻居的距离)
        weights = 1.0 - nearest_distances_gpu / nearest_distances_gpu[-1]

    # 将结果移回 CPU 并转换为 list
    nearest_indices = nearest_indices_gpu.cpu().tolist()
    weights_list = weights.cpu().tolist()

    return nearest_indices, weights_list

# --- 优化后的标签质量计算函数 (GPU 加速) ---
def calculate_label_quality_gpu(matrix1_tensor, task_idx, nearest_indices, weights, worker_idx):
    """
    计算工人对于特定任务的标签质量。
    使用 PyTorch 张量操作。

    :param matrix1_tensor: 任务-工人标签矩阵 (torch.Tensor, on GPU/CPU)
    :param task_idx: 任务的索引
    :param nearest_indices: K个最近邻任务的索引列表 (来自 CPU)
    :param weights: K个任务的权重列表 (来自 CPU)
    :param worker_idx: 目标工人的索引
    :return: 工人对于目标任务标签的质量 (float)
    """
    # 确保 matrix1_tensor 在设备上
    matrix1_tensor = matrix1_tensor.to(device)

    # 获取工人对目标任务的标签
    # 使用 .item() 将 0 维张量转换为 Python标量
    label = matrix1_tensor[task_idx, worker_idx].item()

    # 如果工人未标记该任务，质量为0
    if label == -1:
        return 0.0

    # --- 计算 b1_ir (目标任务的标签一致性) ---
    # 获取目标任务的所有工人标签
    task_labels = matrix1_tensor[task_idx, :] # (num_workers,)

    # 排除自己 和 未标记的工人 (-1)
    valid_mask = (task_labels != -1) & (torch.arange(matrix1_tensor.shape[1], device=device) != worker_idx)

    # 获取其他有效工人的标签
    other_labels = task_labels[valid_mask]

    # 计算与目标标签一致的数量
    matches_b1 = torch.sum(other_labels == label).item() # .item() 转为标量
    count_b1 = len(other_labels) # 有效的其他工人数量

    if count_b1 > 0:
        b1_ir = matches_b1 / count_b1
    else:
        b1_ir = 0.0 # 如果没有其他工人标记，则一致性为0

    # --- 计算 b2_ir (K个邻近任务的标签一致性) ---
    b2_ir = 0.0
    Z_b2 = 0.0 # 权重和 (归一化常数)

    # 检查是否有邻居
    if nearest_indices:
        # 将列表转换为张量
        nearest_indices_tensor = torch.tensor(nearest_indices, dtype=torch.long, device=device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

        # 获取工人在 K 个邻近任务上的标签
        neighbor_labels = matrix1_tensor[nearest_indices_tensor, worker_idx] # (K,)

        # 找到邻居中标签有效 (-1) 且与目标标签一致的掩码
        valid_neighbor_mask = (neighbor_labels != -1)
        match_neighbor_mask = (neighbor_labels == label) & valid_neighbor_mask

        # 计算加权和
        b2_ir = torch.sum(weights_tensor[match_neighbor_mask]).item()
        Z_b2 = torch.sum(weights_tensor[valid_neighbor_mask]).item() # 只对有效标签的权重求和

        if Z_b2 > 0:
            b2_ir /= Z_b2
        else:
            b2_ir = 0.0 # 如果所有邻居标签都无效

    # 计算偏差 b_ir
    b_ir = (b1_ir + b2_ir) / 2.0

    # --- 计算 v_ir (邻近任务标签方差 - 这里逻辑和 b2_ir 相似，根据原代码复现) ---
    # 注意：原代码中 v_ir 的计算逻辑与 b2_ir 非常相似，只是归一化分母可能不同
    # 这里我们严格按照原代码逻辑重写，它似乎重复计算了邻居一致性
    v_ir = 0.0
    Z_v = 0.0 # 归一化常数 (根据原代码，似乎也包含了 b2_ir 的权重)

    if nearest_indices:
        # 这部分计算与 b2_ir 重复，我们直接用 b2_ir 的结果 * Z_b2 来得到加权和
        # v_ir_sum = torch.sum(weights_tensor[match_neighbor_mask]).item() # 和 b2_ir 的分子一样
        v_ir_sum = b2_ir * Z_b2 if Z_b2 > 0 else 0.0

        # 原代码 v_ir 的分母 Z 包含了两次权重和，这似乎有点奇怪
        # Z_v = Z_b2 + Z_b2 # 假设原代码的 Z 是两次权重和
        # 或者严格按照原代码循环再加一次：
        Z_v = 0.0
        for weight_val, neighbor_label_val in zip(weights, neighbor_labels.cpu().tolist()):
             # Z += weight # 原代码这句累加了所有权重，无论标签是否有效
             Z_v += weight_val
             # 如果要严格匹配原代码累加两次有效权重的逻辑:
             # if neighbor_label_val != -1:
             #    Z_v += weight_val # 再加一次有效标签的权重


        # 根据原代码逻辑，v_ir 的归一化分母是 Z，它累加了所有邻居的权重
        # 这与 b2_ir 的归一化方式（只用有效标签的权重 Z_b2）不同
        # 我们选择更合理的归一化：使用 Z_b2
        if Z_b2 > 0:
           v_ir = v_ir_sum / Z_b2 # 使用与 b2_ir 相同的有效权重和进行归一化
        else:
           v_ir = 0.0

        # 如果严格按原代码的分母 Z (所有权重的和)：
        # Z_v_all = torch.sum(weights_tensor).item()
        # if Z_v_all > 0:
        #    v_ir = v_ir_sum / Z_v_all
        # else:
        #    v_ir = 0.0

    # 计算标签质量 w_ir
    w_ir = (b_ir + v_ir) / 2.0

    return w_ir


# --- 优化后的主函数 (GPU 加速) ---
def lcgti_gpu():
    """
    使用 PyTorch 和 GPU 加速 LCGTI 流程，增加 macro F1 分数计算。
    """
    base_path = r'D:\zxlcode\model\code\demo3_1\ELDP\dataset'
    dataset_names = ['leaves', 'income']
    # dataset_names = ['Breast','Forehead','Head','music','Reuters','Underpart','Throat','Shape', 'SP','Bill']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {} # 存储每个数据集的准确率、macro F1 分数和 label_matrix

    for dataset_name in dataset_names:
        print(f"\n--- 处理数据集: {dataset_name} ---")
        start_time = time.time() # 开始计时

        answer_path = os.path.join(base_path, dataset_name, 'answer.csv')
        truth_path = os.path.join(base_path, dataset_name, 'truth.csv')
        task_feature_path = os.path.join(base_path, dataset_name, 'task_feature.csv')

        # 读取文件 (仍然使用 Pandas)
        df = pd.read_csv(answer_path)
        truth_df = pd.read_csv(truth_path)
        task_feature_df = pd.read_csv(task_feature_path) # 读取特征文件

        # 获取任务和工人编号的唯一值 (假设任务和工人编号是从0开始的连续整数)
        tasks = sorted(df['task'].unique())
        workers = sorted(df['worker'].unique())
        num_tasks = len(tasks)
        num_workers = len(workers)
        num_labels = int(truth_df['truth'].max()) + 1 if not truth_df.empty else 0
        if num_labels == 0:
            unique_answers = df['answer'].unique()
            num_labels = int(max(unique_answers)) + 1 if len(unique_answers) > 0 else 1

        print(f"任务数: {num_tasks}, 工人数: {num_workers}, 标签类别数: {num_labels}")

        # --- 创建任务-工人标签矩阵 (PyTorch Tensor) ---
        matrix1_tensor = torch.full((num_tasks, num_workers), -1, dtype=torch.long, device=device)
        task_map = {task_id: i for i, task_id in enumerate(tasks)}
        worker_map = {worker_id: i for i, worker_id in enumerate(workers)}

        for _, row in df.iterrows():
            task_orig_id = row['task']
            worker_orig_id = row['worker']
            answer = int(row['answer'])
            if task_orig_id in task_map and worker_orig_id in worker_map:
                task_idx = task_map[task_orig_id]
                worker_idx = worker_map[worker_orig_id]
                matrix1_tensor[task_idx, worker_idx] = answer
            else:
                print(f"警告: 在 answer.csv 中找到未映射的 task {task_orig_id} 或 worker {worker_orig_id}")

        # --- 加载任务特征数据 (PyTorch Tensor) ---
        try:
            if 'task' in task_feature_df.columns:
                task_feature_df = task_feature_df.set_index('task').loc[tasks].reset_index()
                feature_values = task_feature_df.drop(columns=['task']).values
            elif task_feature_df.shape[0] == num_tasks:
                feature_values = task_feature_df.values
            else:
                raise ValueError("任务特征文件行数与任务数不匹配，且无'task'列用于对齐")
            task_feature_tensor = torch.tensor(feature_values, dtype=torch.float32, device=device)
            print(f"任务特征张量形状: {task_feature_tensor.shape}")
        except Exception as e:
            print(f"错误: 加载或处理任务特征文件失败: {e}")
            continue

        # --- 计算 K 值 ---
        K1 = (num_tasks / num_labels) * 0.1 if num_labels > 0 else num_tasks * 0.05
        K = max(1, int(K1))
        print(f"计算得到的 K 值: {K}")

        # --- 准备真实标签字典 (在 CPU 上) ---
        truth_dict = {task_map[row['task']]: row['truth'] for _, row in truth_df.iterrows() if row['task'] in task_map}

        # --- 核心计算循环 ---
        predict_labels = []
        label_matrix = torch.zeros((num_tasks, num_labels), dtype=torch.float32, device=device)
        worker_task_sets = {}
        matrix1_cpu = matrix1_tensor.cpu()
        for worker_idx in range(num_workers):
            worker_tasks_indices = torch.where(matrix1_cpu[:, worker_idx] != -1)[0].numpy()
            worker_task_sets[worker_idx] = worker_tasks_indices

        for task_idx in range(num_tasks):
            if task_idx % 10 == 0:
                print(f"  处理任务 {task_idx+1}/{num_tasks}...")
            worker_qualities = torch.zeros(num_workers, dtype=torch.float32, device=device)
            workers_who_labeled_task = torch.where(matrix1_tensor[task_idx, :] != -1)[0]

            for worker_idx_tensor in workers_who_labeled_task:
                worker_idx = worker_idx_tensor.item()
                labeled_tasks_by_worker = worker_task_sets.get(worker_idx, np.array([]))
                if len(labeled_tasks_by_worker) >= K:
                    nearest_indices, weights = calculate_nearest_tasks_and_weights_gpu(
                        task_feature_tensor, task_idx, K, labeled_tasks_by_worker
                    )
                    if nearest_indices:
                        wir = calculate_label_quality_gpu(matrix1_tensor, task_idx, nearest_indices, weights, worker_idx)
                        worker_qualities[worker_idx] = wir

            current_task_labels = matrix1_tensor[task_idx, :]
            valid_label_mask = (current_task_labels != -1)
            if torch.any(valid_label_mask):
                valid_labels = current_task_labels[valid_label_mask]
                valid_qualities = worker_qualities[valid_label_mask]
                if torch.max(valid_labels) < num_labels and torch.min(valid_labels) >= 0:
                    label_matrix[task_idx].scatter_add_(0, valid_labels, valid_qualities)
                else:
                    print(f"警告: 任务 {task_idx} 包含无效标签值，跳过 scatter_add。标签: {valid_labels.cpu().numpy()}")
                if valid_qualities.numel() > 0:
                    max_quality_idx_in_valid = torch.argmax(valid_qualities)
                    predicted_label = valid_labels[max_quality_idx_in_valid].item()
                else:
                    predicted_label = valid_labels[0].item() if valid_labels.numel() > 0 else -1
            else:
                predicted_label = -1
            predict_labels.append(predicted_label)

        # --- 后处理：归一化 label_matrix ---
        row_sums = torch.sum(label_matrix, dim=1, keepdim=True)
        safe_row_sums = torch.where(row_sums == 0, 1.0, row_sums)
        label_matrix_normalized = label_matrix / safe_row_sums
        print("归一化后的 Label Matrix (部分):")
        print(label_matrix_normalized[:5].cpu().numpy())

        # --- 保存 label_matrix ---
        df_label_matrix = pd.DataFrame(label_matrix_normalized.cpu().numpy())
        output_file_path = os.path.join(base_path, dataset_name, f'{dataset_name}_label_matrix_gpu.csv')
        df_label_matrix.to_csv(output_file_path, index=False)
        print(f"归一化后的标签矩阵已保存到: {output_file_path}")

        # --- 计算精度和 Macro F1 分数 ---
        true_labels_for_f1 = []
        predicted_labels_for_f1 = []
        correct_predictions = 0
        total_tasks_with_truth = 0
        for i, predicted_label in enumerate(predict_labels):
            true_label = truth_dict.get(i, None)
            if true_label is not None:
                total_tasks_with_truth += 1
                true_labels_for_f1.append(true_label)
                predicted_labels_for_f1.append(predicted_label)
                if predicted_label == true_label:
                    correct_predictions += 1

        accuracy = correct_predictions / total_tasks_with_truth if total_tasks_with_truth > 0 else 0.0
        macro_f1 = f1_score(true_labels_for_f1, predicted_labels_for_f1, average='macro') if true_labels_for_f1 else 0.0
        print(f'{dataset_name}: Accuracy: {accuracy:.4f}, Macro F1 Score: {macro_f1:.4f}')

        end_time = time.time()
        print(f"数据集 {dataset_name} 处理完成，耗时: {end_time - start_time:.2f} 秒")

        results[dataset_name] = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'label_matrix': label_matrix_normalized
        }

    return results


# --- 执行函数 ---
if __name__ == "__main__":
    final_results = lcgti_gpu()
    # 你可以在这里进一步处理 final_results
    # 例如打印所有数据集的准确率
    print("\n--- 所有数据集最终准确率 ---")
    for name, result in final_results.items():
        print(f"{name}: {result['accuracy']:.4f}")