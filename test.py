import numpy as np


def create_gaussian_distributed_list(center_idx, num_elements, std_dev=1.0):
    # 创建高斯分布
    x = np.linspace(-3, 3, num_elements)  # 生成 num_elements 个点，范围从 -3 到 3
    gaussian = np.exp(-x ** 2 / (2 * std_dev ** 2))  # 计算高斯分布值

    # 将高斯分布的中心移动到指定的中心索引
    shift = (num_elements // 2) - center_idx
    gaussian = np.roll(gaussian, shift)

    # 归一化，使最大值为1
    gaussian = gaussian / np.max(gaussian)

    return gaussian


# 定义one-hot列表，其中包含16个元素
one_hot_list = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
center_idx = one_hot_list.index(1)  # 找到元素1的位置

# 生成高斯分布的列表
gaussian_list = create_gaussian_distributed_list(center_idx, 16)

# 复制每个元素并放在右边相邻位置，形成一个长度为32的列表
duplicated_list = []
for value in gaussian_list:
    duplicated_list.append(value)
    duplicated_list.append(value)

print(duplicated_list)
