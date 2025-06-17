import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from KNN_and_weight import calculate_nearest_tasks_and_weights
from set_w import calculate_label_quality
import os


# 读取 label.csv 文件
# df = pd.read_csv(r'D:\model\code\dataset\Reuters\label.csv')
# truth_df = pd.read_csv(r'D:\model\code\dataset\Reuters\truth.csv')  # 请根据实际路径修改
# file_path = r'D:\model\code\dataset\Reuters\task_feature（bad）.csv'

def lcgti():
    # 基础路径，包含所有数据集的文件夹
    base_path = r'D:\zxlcode\model\code\demo3_1\ELDP\dataset'
    # 需要处理的数据集名称
    # dataset_names = ['Breast','Forehead','Head','Music','Reuters','Underpart','Throat','Shape', 'SP','Bill']
    # dataset_names = ['leaves16']
    dataset_names = ['leaves', 'income']


    # 逐个读取指定的数据集
    for dataset_name in dataset_names:
        # 构建文件路径
        answer_path = os.path.join(base_path, dataset_name, 'answer.csv')
        truth_path = os.path.join(base_path, dataset_name, 'truth.csv')
        task_feature_path = os.path.join(base_path, dataset_name, 'task_feature.csv')

        # 读取文件
        df = pd.read_csv(answer_path)

        truth_df = pd.read_csv(truth_path)

        file_path = task_feature_path

        # 这里可以进行后续的数据处理，模型训练等操作

        # 获取任务和工人编号的唯一值
        tasks = df['task'].unique()  # 所有任务
        workers = df['worker'].unique()  # 所有工人
        num_tasks = df['task'].nunique()  # 任务数
        num_labels = df['answer'].nunique()  # 标签个数
        # 初始化一个任务 x 工人矩阵，初始值为 -1，表示未回答
        matrix1 = pd.DataFrame(-1, index=tasks, columns=workers, dtype=int)

        # 填充矩阵1
        for _, row in df.iterrows():
            task = row['task']
            worker = row['worker']
            answer = row['answer']
            matrix1.at[task, worker] = answer  # 用实际的回答填充矩阵

        # 处理任务并计算与任务最相似的K个任务
        def process_task(file_path, task_idx, matrix1, K):
            # 针对这个任务的工人质量集合
            w = []
            # 获取任务对应的工人回答
            task_answers = matrix1.iloc[task_idx, :].values  # 获取任务idx的回答（工人1，工人2，...）
            # 统计每个工人回答的任务数量（即该工人回答过的任务数，不为-1）
            worker_task_counts = np.sum(matrix1 != -1, axis=0)  # 按列统计，每列代表一个工人的回答数

            # 遍历任务的工人回答数据
            # 获取每个工人回答过的任务编号集合
            worker_task_sets = {}
            for worker_idx in range(matrix1.shape[1]):  # 遍历每个工人
                worker_tasks = np.where(matrix1.iloc[:, worker_idx] != -1)[0]  # 找到该工人回答过的任务编号
                worker_task_sets[worker_idx] = worker_tasks

            # 遍历任务的工人回答数据
            valid_workers = []
            for worker_idx, task_count in enumerate(worker_task_counts):
                # 针对这个任务的工人质量集合
                # print('worker_id:',worker_idx)
                # print('task_count:', task_count)
                # 如果工人回答过的任务数大于等于K，则认为该工人有效
                if task_count >= K:
                    valid_workers.append(worker_idx)
                    nearest_indices, weights = calculate_nearest_tasks_and_weights(file_path, task_idx, K,
                                                                                   worker_task_sets[worker_idx])
                    wir = calculate_label_quality(matrix1, task_idx, nearest_indices, weights,
                                                  worker_idx)  # 计算worker_idx这工人对task_idx这任务的标签质量
                else:
                    wir = 0
                # print('wir:', wir)
                w.append(wir)
            return w

        truth_dict = dict(zip(truth_df['task'], truth_df['truth']))  # 创建一个字典方便快速查找真实标签
        predict_labels = []

        K1 = (num_tasks / num_labels) * 0.1
        K = int(K1)
        print(K)
        sum_acc = 0

        label_matrix = np.zeros((num_tasks, num_labels))

        for index in range(num_tasks):
            w = process_task(file_path, index, matrix1, K)
            print(w)
            max_weight = max(w)
            worker_index = w.index(max_weight)
            predict_label = matrix1.iloc[index, worker_index]
            predict_labels.append(predict_label)
            # 将当前任务的每个工人的标签及其对应的权重累加到 label_matrix 中
            for worker_idx, worker_weight in enumerate(w):
                # 获取工人对该任务的标签
                label = matrix1.iloc[index, worker_idx]

                # 累加权重到 label_matrix 的对应列
                label_matrix[index, label] += worker_weight
        # 对 label_matrix 的每一行进行归一化处理
        for i in range(num_tasks):
            row_sum = np.sum(label_matrix[i, :])
            if row_sum != 0:
                label_matrix[i, :] /= row_sum  # 归一化处理，使每行的权重和为 1
        print(label_matrix)
        df1 = pd.DataFrame(label_matrix)
        # file_path = f'D:\\zxlcode\\model\\code\\dataset\\{dataset_name}\\{dataset_name}.csv'
        # df1.to_csv(file_path, index=False)
        correct_predictions = 0
        for i, predicted_label in enumerate(predict_labels):
            task_id = i  # 任务编号与任务的顺序一致
            true_label = truth_dict.get(task_id, None)  # 获取真实标签
            if true_label is not None and predicted_label == true_label:
                correct_predictions += 1



            accuracy = correct_predictions / num_tasks if num_tasks > 0 else 0
            print(f'{dataset_name}: Accuracy: {accuracy:.4f}')

            # 计算 macro F1
            y_true = [truth_dict.get(i) for i in range(num_tasks)]
            y_pred = predict_labels
            macro_f1 = f1_score(y_true, y_pred, average='macro')
            print(f'{dataset_name}: Macro F1 Score: {macro_f1:.4f}')

    return label_matrix


lcgti()
