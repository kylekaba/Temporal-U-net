import os
import cv2
import numpy as np

# This code is used to calculate the MSE between the paired images
# (training and testing need to be calculated separately)
folder1 = './data/T-CLOUD/train/reference'
folder2 = './data/T-CLOUD/train/cloud'


# 获取文件夹中的所有图片文件
image_files1 = sorted(
    [os.path.join(folder1, filename) for filename in os.listdir(folder1) if filename.endswith('.png')])
image_files2 = sorted(
    [os.path.join(folder2, filename) for filename in os.listdir(folder2) if filename.endswith('.png')])

# 初始化MSE的最大和最小值
max_mse = float('-inf')
min_mse = float('inf')

# 遍历对应的图像文件，计算MSE
for file1, file2 in zip(image_files1, image_files2):
    image1 = cv2.imread(file1)
    image2 = cv2.imread(file2)

    # 将图像转换为浮点数并进行归一化
    normalized_image1 = image1.astype(np.float32) / 255.0
    normalized_image2 = image2.astype(np.float32) / 255.0

    mse = np.mean(np.square(normalized_image1 - normalized_image2))

    max_mse = max(max_mse, mse)
    min_mse = min(min_mse, mse)

print("最大MSE值:", max_mse)
print("最小MSE值:", min_mse)

