import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import sys
import glob
import hdf5storage
from random import shuffle
import time
import os
import cv2
import os
import h5py
import re
import config
import connections
import coordinates
import estimators
from getrideof import *
stride = 8
#mat_file = 'Test/jump_009-frame049.mat'
#mat_file = 'Test/push_009-frame058.mat'
#mat_file = 'C:/Users\86183\Downloads\Wi-Pose\Wi-Pose\Train/wave_118-frame095.mat'

#wisppn = torch.load('wisppn-20241016-epoch99.pkl')
def plot_skeleton(skeleton_coords):
    # 将一维数组分割为x和y坐标
    y_coords = skeleton_coords[:18]
    x_coords = skeleton_coords[18:36]
    #print('错误坐标',y_coords[17],x_coords[17])



    # 创建一个图形和一个坐标轴
    fig, ax = plt.subplots()

    # 绘制每个骨骼点
    #ax.plot(x_coords, y_coords, 'o')
    # 绘制每个骨骼点并标注索引
    for i in range(len(x_coords)):
        ax.plot(x_coords[i], y_coords[i], 'o')
        ax.text(x_coords[i], y_coords[i], str(i), fontsize=9, ha='right')
        #print('坐标',x_coords[i],y_coords[i])

    # 定义人体骨骼的连接方式，这里需要根据实际的骨骼连接来定义
    # 例如，以下是一些假设的连接，你需要根据实际情况调整
    connections = [
        (0, 1),  # 鼻子到脖子
        (1, 2), (2, 3), (3, 4),  # 脖子到右肩到右肘到右腕
        (1, 5), (5, 6), (6, 7),  # 脖子到左肩到左肘到左腕
        (1, 8), (8, 9), (9, 10),  # 脖子到右髋到右膝到右踝
        (1, 11), (11, 12), (12, 13),  # 脖子到左髋到左膝到左踝
        (14, 15),  # 鼻子到右眼到左眼
        (0, 16) ,(0,17) # 鼻子到右耳到左耳
    ]
    # 绘制骨骼连接线
    for connection in connections:
        start_point = connection[0]
        end_point = connection[1]
        ax.plot([x_coords[start_point], x_coords[end_point]], [y_coords[start_point], y_coords[end_point]], 'k-')

    # 设置坐标轴的比例相等以保持图形不变形
    ax.set_aspect('equal')
    # 将y轴方向翻转，使得原点在左上角
    ax.invert_yaxis()

    # 隐藏坐标轴
    ax.axis('off')

    # 显示图例
    ax.legend()

    # 显示图形
    plt.show()

def doubleplot(true_x,true_y,predicted_x,predicted_y):
    # 定义人体骨骼的连接方式
    connections = [
        (0, 1),  # 鼻子到脖子
        (1, 2), (2, 3), (3, 4),  # 脖子到右肩到右肘到右腕
        (1, 5), (5, 6), (6, 7),  # 脖子到左肩到左肘到左腕
        (1, 8), (8, 9), (9, 10),  # 脖子到右髋到右膝到右踝
        (1, 11), (11, 12), (12, 13),  # 脖子到左髋到左膝到左踝
        (14, 15),  # 鼻子到右眼到左眼
        (0, 16), (0, 17)  # 鼻子到右耳到左耳
    ]
    # 创建一个图形和两个坐标轴
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    y_coords=true_x
    x_coords=true_y

    # 绘制每个骨骼点并标注索引
    for i in range(len(x_coords)):
        ax1.plot(x_coords[i], y_coords[i], 'o')
        ax1.text(x_coords[i], y_coords[i], str(i), fontsize=9, ha='right')
        # print('坐标',x_coords[i],y_coords[i])


    # 绘制骨骼连接线
    for connection in connections:
        start_point = connection[0]
        end_point = connection[1]
        ax1.plot([x_coords[start_point], x_coords[end_point]], [y_coords[start_point], y_coords[end_point]], 'k-')
    # 设置坐标轴的比例相等以保持图形不变形
    ax1.set_aspect('equal')
    # 将y轴方向翻转，使得原点在左上角
    ax1.invert_yaxis()
    # 隐藏坐标轴
    ax1.axis('off')
    # 显示图例
    
    y_coords = predicted_x
    x_coords = predicted_y

    for i in range(len(x_coords)):
        ax2.plot(x_coords[i], y_coords[i], 'o')
        ax2.text(x_coords[i], y_coords[i], str(i), fontsize=9, ha='right')
    # 绘制骨骼连接线
    for connection in connections:
        start_point = connection[0]
        end_point = connection[1]
        ax2.plot([x_coords[start_point], x_coords[end_point]], [y_coords[start_point], y_coords[end_point]], 'k-')
    # 设置坐标轴的比例相等以保持图形不变形
    ax2.set_aspect('equal')
    # 将y轴方向翻转，使得原点在左上角
    ax2.invert_yaxis()
    # 隐藏坐标轴
    ax2.axis('off')
    # 显示图例
    
    # 设置子图标题
    ax1.set_title('Ground Truth')
    ax2.set_title('Predict')
    plt.show()


def readwipose(mat_file):
    with h5py.File(mat_file, 'r') as file:
        # 列出所有的数据集名称
        #print(list(file.keys()))

        # 假设您要读取的数据集名称为 'your_dataset_name'
        # 读取数据集并转置（因为MATLAB和Python的数组索引顺序不同）
        CSIdata = np.array(file['CSI'])
        Skeleton = np.array(file['SkeletonPoints'])
        #print(Skeleton)
        CSIdata = CSIdata.T  # 或者使用 np.transpose(data)
        Skeleton = Skeleton.T

        # print(CSIdata[0])
        # print(Skeleton)
    # 假设 data 是你的 54 维 NumPy 数组

    #plot_skeleton(Skeleton[:36])

    # data = np.random.rand(54)  # 示例数据
    adjacency_matrix = create_adjacency_matrix(Skeleton)

    #for index in range(18):
        #print('矩阵坐标', adjacency_matrix[0, index, index], adjacency_matrix[1, index, index])
    #print(adjacency_matrix.shape)  # 应该输出 (4, 18, 18)
    return adjacency_matrix
def readskeleton(mat_file):
    with h5py.File(mat_file, 'r') as file:
        # 列出所有的数据集名称
        print(list(file.keys()))

        # 假设您要读取的数据集名称为 'your_dataset_name'
        # 读取数据集并转置（因为MATLAB和Python的数组索引顺序不同）
        #CSIdata = np.array(file['CSI'])
        Skeleton = np.array(file['SkeletonPoints'])
        #print(Skeleton)
        #CSIdata = CSIdata.T  # 或者使用 np.transpose(data)
        Skeleton = Skeleton.T

        # print(CSIdata[0])
        # print(Skeleton)
    # 假设 data 是你的 54 维 NumPy 数组

    #plot_skeleton(Skeleton[:36])


    return Skeleton

def create_adjacency_matrix(data):
    # 初始化输出数组
    output = np.zeros((4, 18, 18))

    # 获取 x, y 坐标和概率值
    x_coords = data[:18]
    y_coords = data[18:36]
    probabilities = data[36:]

    # 构建 x 和 y 坐标的邻接矩阵
    for i in range(18):
        for j in range(18):
            if i == j:
                output[0, i, j] = x_coords[i]
                output[1, i, j] = y_coords[i]
            else:
                output[0, i, j] = x_coords[i] - x_coords[j]
                output[1, i, j] = y_coords[i] - y_coords[j]

    # 构建概率值的邻接矩阵
    for i in range(18):
        for j in range(18):
            if i == j:
                output[2, i, j] = probabilities[i]
                output[3, i, j] = probabilities[i]
            else:
                output[2, i, j] = probabilities[i] * probabilities[j]
                output[3, i, j] = probabilities[i] * probabilities[j]

    # 第三个通道（索引为 3）在这个例子中没有使用，所以保持为 0
    # 如果需要，可以在这里添加额外的逻辑来填充这个通道

    return output

def matixplot(pred_xy):
    # 这��我们取第一个样本的预测结果进行显示
    pred_xy_sample = pred_xy

    # 创建一个图形窗口，包含两个子图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 在第一个子图上绘制第一个通道的热图
    axs[0].imshow(pred_xy_sample[0, :, :], cmap='hot', interpolation='nearest')
    axs[0].set_title('Channel 0 Heatmap')

    # 在第二个子图上绘制第二个通道的热图
    axs[1].imshow(pred_xy_sample[1, :, :], cmap='hot', interpolation='nearest')
    axs[1].set_title('Channel 1 Heatmap')

    # 为整个图形窗口添加一个标题
    fig.suptitle('18x18 Matrix Heatmaps')

    # 显示图形窗口
    plt.show()


def doubleplot(true_x,true_y,predicted_x,predicted_y):
    # 定义人体骨骼的连接方式
    connections = [
        (0, 1),  # 鼻子到脖子
        (1, 2), (2, 3), (3, 4),  # 脖子到右肩到右肘到右腕
        (1, 5), (5, 6), (6, 7),  # 脖子到左肩到左肘到左腕
        (1, 8), (8, 9), (9, 10),  # 脖子到右髋到右膝到右踝
        (1, 11), (11, 12), (12, 13),  # 脖子到左髋到左膝到左踝
        (14, 15),  # 鼻子到右眼到左眼
        (0, 16), (0, 17)  # 鼻子到右耳到左耳
    ]
    # 创建一个图形和两个坐标轴
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    y_coords=true_x
    x_coords=true_y

    # 绘制每个骨骼点并标注索引
    for i in range(len(x_coords)):
        ax1.plot(x_coords[i], y_coords[i], 'o')
        ax1.text(x_coords[i], y_coords[i], str(i), fontsize=9, ha='right')
        # print('坐标',x_coords[i],y_coords[i])


    # 绘制骨骼连接线
    for connection in connections:
        start_point = connection[0]
        end_point = connection[1]
        ax1.plot([x_coords[start_point], x_coords[end_point]], [y_coords[start_point], y_coords[end_point]], 'k-')
    # 设置坐标轴的比例相等以保持图形不变形
    ax1.set_aspect('equal')
    # 将y轴方向翻转，使得原点在左上角
    ax1.invert_yaxis()
    # 隐藏坐标轴
    ax1.axis('off')
    # 显示图例
    
    y_coords = predicted_x
    x_coords = predicted_y

    for i in range(len(x_coords)):
        ax2.plot(x_coords[i], y_coords[i], 'o')
        ax2.text(x_coords[i], y_coords[i], str(i), fontsize=9, ha='right')
    # 绘制骨骼连接线
    for connection in connections:
        start_point = connection[0]
        end_point = connection[1]
        ax2.plot([x_coords[start_point], x_coords[end_point]], [y_coords[start_point], y_coords[end_point]], 'k-')
    # 设置坐标轴的比例相等以保持图形不变形
    ax2.set_aspect('equal')
    # 将y轴方向翻转，使得原点在左上角
    ax2.invert_yaxis()
    # 隐藏坐标轴
    ax2.axis('off')
    # 显示图例
    
    # 设置子图标题
    ax1.set_title('Ground Truth')
    ax2.set_title('Predict')
    plt.show()


def calculate_l2_loss(pred, target, confidence):
    """
    计算两个2x18x18的numpy数组之间的L2损失，考虑置信度。
    参数:
    - pred: 预测坐标的numpy数组，形状为(2, 18, 18)。
    - target: 真实坐标的numpy数组，形状为(2, 18, 18)。
    - confidence: 关键点的置信度numpy数组，形状为(2, 18, 18)。

    返回:
    - l2_loss: 计算得到的L2损失。
    """
    # 确保输入数组是浮点类型
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    confidence = confidence.astype(np.float32)
    # 计算加权差异
    weighted_diff = confidence * (pred - target)
    # 平方差异
    squared_diff = np.square(weighted_diff)
    # 求和
    sum_squared_diff = np.sum(squared_diff)
    # 计算平均损失
    l2_loss = sum_squared_diff / (2 * 18 * 18)
    return l2_loss

import numpy as np
def heatplot(heatmap_avg1,paf_avg1):
    heatmap_avg_mean = np.max(heatmap_avg1, axis=2)  # 沿着第三个维度（通道）计算平均值

    # 计算paf的平均值
    paf_avg_mean = np.max(paf_avg1, axis=2)  # 沿着第三个维度（通道）计算平均值
    heatmap_avg_mean_255 = cv2.normalize(heatmap_avg_mean, None, 0, 255, cv2.NORM_MINMAX)
    paf_avg_mean_255 = cv2.normalize(paf_avg_mean, None, 0, 255, cv2.NORM_MINMAX)

    # 将平均值转换为uint8类型，以便使用OpenCV进行显示
    heatmap_avg_mean_255 = heatmap_avg_mean_255.astype(np.uint8)
    paf_avg_mean_255 = paf_avg_mean_255.astype(np.uint8)
    cv2.namedWindow('Heatmap and PAF', cv2.WINDOW_NORMAL)

    # 显示heatmap的平均热力图
    cv2.imshow('Heatmap and PAF', cv2.cvtColor(heatmap_avg_mean_255, cv2.COLOR_GRAY2BGR))
    cv2.waitKey(1)  # 添加这行很重要！
    #cv2.imshow('Heatmap and PAF', cv2.cvtColor(paf_avg_mean_255, cv2.COLOR_GRAY2BGR))
    #cv2.waitKey(1)  # 添加这行很重要！
    '''
    import matplotlib.pyplot as plt

    # 使用matplotlib显示热力图
    plt.figure(figsize=(10, 5))

    # 显示heatmap的平均热力图
    plt.subplot(1, 2, 1)
    plt.imshow(heatmap_avg_mean_255, cmap='jet')
    plt.title('Heatmap Average')

    # 显示paf的平均热力图
    plt.subplot(1, 2, 2)
    plt.imshow(paf_avg_mean_255, cmap='jet')
    plt.title('PAF Average')

    plt.show()
'''
    


def calculate_prediction_error(true_x, true_y, predicted_x, predicted_y):
    """
    Calculate the prediction error between true and predicted coordinates.
    
    Args:
    true_x (np.array): True x coordinates
    true_y (np.array): True y coordinates
    predicted_x (np.array): Predicted x coordinates
    predicted_y (np.array): Predicted y coordinates
    
    Returns:
    dict: A dictionary containing MSE, MAE, and PCK for x and y coordinates
    """
    # Ensure all inputs are numpy arrays
    true_x = np.array(true_x)
    true_y = np.array(true_y)
    predicted_x = np.array(predicted_x)
    predicted_y = np.array(predicted_y)
    
    # Calculate errors for x coordinates
    mse_x = np.mean((true_x - predicted_x) ** 2)
    mae_x = np.mean(np.abs(true_x - predicted_x))
    
    # Calculate errors for y coordinates
    mse_y = np.mean((true_y - predicted_y) ** 2)
    mae_y = np.mean(np.abs(true_y - predicted_y))
    
    # Calculate overall errors
    mse_overall = (mse_x + mse_y) / 2
    mae_overall = (mae_x + mae_y) / 2
    
    # Calculate reference distance (between shoulder and hip)
    ref_dist = np.sqrt((true_x[5] - true_x[8])**2 + (true_y[5] - true_y[8])**2)
    
    # Calculate PCK for different thresholds
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    pck_results = {}
    
    for i in range(len(true_x)):
        distance = np.sqrt((true_x[i] - predicted_x[i])**2 + (true_y[i] - predicted_y[i])**2)
        for threshold in thresholds:
            distance_threshold = ref_dist * threshold
            pck = 1 if distance <= distance_threshold else 0
            pck_results[f'PCK_{i}@{int(threshold*100)}'] = pck
    
    # Calculate average PCK for each threshold
    for key in pck_results:
        pck_results[key] = np.mean(pck_results[key])
    
    return {
        'MSE_X': mse_x,
        'MAE_X': mae_x,
        'MSE_Y': mse_y,
        'MAE_Y': mae_y,
        'MSE_Overall': mse_overall,
        'MAE_Overall': mae_overall,
        **pck_results
    }

def evalue_all(mat_file,wisppn):
    #mat_file = 'C:/Users\86183\Downloads\Wi-Pose\Wi-Pose\Train/wave_118-frame082.mat'
    #wisppn = torch.load('wisppn-20241021.pkl')
    #GT=readwipose(mat_file)
    


    

    csi_data = torch.zeros(1, 3, 30, 30)
    GT_heatmap_st1 = torch.zeros(1, 6, 36, 36)
    GT_heatmap_st2 = torch.zeros(1, 6, 36, 36)
    GT_heatmap_st3 = torch.zeros(1, 7, 36, 36)

    csi_data[0,:,:,:] = torch.from_numpy(PracticalCSIInput(mat_file)).type(torch.FloatTensor)
    _, GT_heatmap_st1[0, :, :, :] = OppMatProcess_ST1(mat_file)
    _, GT_heatmap_st2[0, :, :, :] = OppMatProcess_ST2(mat_file)
    _, GT_heatmap_st3[0, :, :, :] = OppMatProcess_ST3(mat_file)


    wisppn = wisppn.cuda().eval()

    csi_data = Variable(csi_data.cuda())
    #print(csi_data.shape)
    Mconv7_stage6_L1,Mconv7_stage6_L2,paf3,pcm3,paf2,pcm2,paf1,pcm1 = wisppn(csi_data,GT_heatmap_st1,GT_heatmap_st1)
    #mat_file2 = 'newtrain/3.mat'
    Mconv7_stage6_L2= PcmReconstruct(pcm1,pcm2,pcm3)

    #Mconv7_stage6_L2 = pcm4
    #Mconv7_stage6_L1 = paf4
    #Mconv7_stage6_L1 = Mconv7_stage6_L1.cpu().detach().numpy()
    Mconv7_stage6_L2 = Mconv7_stage6_L2.cpu().detach().numpy()
    heatmap = np.transpose(np.squeeze(Mconv7_stage6_L2), (1, 2, 0))
    heatmap = cv2.resize(heatmap, (0,0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
    heatmap = heatmap[:294, :400, :]  # 去除填充
    heatmap = cv2.resize(heatmap, (640, 480), interpolation=cv2.INTER_CUBIC)

  

    
    thre1 = 0.05
    thre2 = 0.05

    
    cfg = config.get_default_configuration()
    # 得到各个身体部分的坐标和得分情况，一共有18个人体关键点。(x ,y, score, id)
    coords = coordinates.get_coordinates(cfg, heatmap, thre1)
 
    predicted_x = []
    predicted_y = []

    # 遍历字典提取坐标
    for key, value in coords.items():
        if value:  # 检查列表是否非空
            x, y, *_ = value[0]  # 提取第一个元素的x和y坐标
        else:
            x, y = 0, 0  # 空列表时，记录为0
        predicted_x.append(x)
        predicted_y.append(y)

    
    # 打印结果
    
    #mat_file2 = 'newtrain/3.mat'
    
    true_x, true_y = readOppGT(mat_file)
    

    errors= calculate_prediction_error(true_x, true_y, predicted_x, predicted_y)
    print('predicted_x:', predicted_x)
    print('标签值',true_x)
    print('predicted_y:', predicted_y)
    print('标签值',true_y)
    #doubleplot(true_y,true_x,predicted_y,predicted_x)

    #绘制推理矩阵
    #matixplot(pred_xy[0])
    #绘制GT矩阵
    #matixplot(GT)

   
    return errors
    

def evalue_all_batch(mats,wisppn):
    all_errors = {}
    
    for mat in mats:
        #print('processing',mat)
        errors = evalue_all(mat, wisppn)
        
        
        # 初始化 all_errors 字典
        if not all_errors:
            all_errors = {key: [] for key in errors.keys()}
        
        # 累积每个 mat 文件的误差
        for key, value in errors.items():
            all_errors[key].append(value)
    
    # 计算平均误差
    average_errors = {key: np.mean(values) for key, values in all_errors.items()}
    
    # 打印平均误差
    #print("Average errors across all .mat files:")
    #for key, value in average_errors.items():
        #print(f"{key}: {value}")
    # 对相同 alpha 的 PCK 求平均
    pck_pattern = re.compile(r'PCK_(\d+)@(\d+)')
    pck_by_alpha = {}
    
    for key, value in average_errors.items():
        match = pck_pattern.match(key)
        if match:
            alpha = int(match.group(2))
            if alpha not in pck_by_alpha:
                pck_by_alpha[alpha] = []
            pck_by_alpha[alpha].append(value)
    
    average_pck_by_alpha = {f'PCK@{alpha}': np.mean(values) for alpha, values in pck_by_alpha.items()}
    
    # 打印平均误差
    print("Average errors across all .mat files:")
    for key, value in average_errors.items():
        if not key.startswith('PCK_'):
            print(f"{key}: {value}")
    
    print("\nAverage PCK by alpha:")
    for key, value in average_pck_by_alpha.items():
        print(f"{key}: {value}")
    


    
if __name__=="__main__":
    # 创建一个窗口
    
    
    #mat_file = 'C:/Users\86183\Downloads\Wi-Pose\Wi-Pose\Train/wave_118-frame082.mat'
    #wisppn = torch.load('weights/wisppn-20241027-epoch199.pkl')
    wisppn = torch.load('weights/epochsave.pkl')
    
    #mat_file = 'test.mat'
    #evalue_all(mat_file,wisppn)
    mats = []
    mats += glob.glob('/mnt/workspace/group2_30_percent2/*.mat')
    mats += glob.glob('/mnt/workspace/group2_30_percent1/*.mat')
    #mats = glob.glob('C:/Users\86183\Desktop/result2/result2/*.mat')
    import random
    #mats=mats[30:720:47]
    mats = random.sample(mats, 10)
    #print(mats)
    evalue_all_batch(mats,wisppn)
    #mats = glob.glob('C:/Users\86183\Desktop/result2/validate/*.mat')
    #mats = glob.glob('C:/Users\86183\Desktop/result2/result2/*.mat')
    
    #mats=mats[30:720:47]
    #mats=mats[1:1000:2]
    #print(mats)
   # evalue_all_batch(mats,wisppn)
    #evalue_all(mat_file,wisppn)

    