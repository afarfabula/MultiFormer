import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
from scipy.io import loadmat
from scipy.io import savemat
#import seaborn as sns
from scipy.interpolate import interp1d
import cv2
from scipy.ndimage import zoom
from scipy.signal import firwin, resample_poly

def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

def judgedis(mat_file):
    with h5py.File(mat_file, 'r') as file:
        Skeleton = np.array(file['SkeletonPoints'])
        Skeleton = Skeleton.T

        n_point = [Skeleton[0], Skeleton[18]]
        ear_point = [Skeleton[17], Skeleton[35]]
        distance = calculate_distance(n_point, ear_point)

    return distance

# 获取所有.mat文件的路径
mats = glob.glob('Test/*.mat')

def distri_plot():
    # 只取前100个文件路径
    mats_little = mats[:3000]

    # 存储每个文件的距离
    distances = []

    # 遍历每个文件并计算距离
    for mat_file in mats_little:
        distance = judgedis(mat_file)
        distances.append(distance)

    # 确保distances是一个一维列表
    distances = [distance for sublist in distances for distance in sublist]

    # 绘制直方图
    plt.hist(distances, bins=20, alpha=0.7, color='blue')
    plt.title('Distribution of Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.show()
#distri_plot()


def delete_unnormal(mats):
    # 遍历每个文件并计算距离
    for mat_file in mats:
        try:
            distance = judgedis(mat_file)
            if distance > 80:
                # 删除该文件
                os.remove(mat_file)
                print(f"Deleted {mat_file} due to abnormal distance.")
        except Exception as e:
            # 如果在处理文件时发生异常，打印错误信息
            print(f"An error occurred while processing {mat_file}: {e}")

# 示例使用
def maketest():
    import os
    import random
    import shutil

    # 定义源目录和目标目录
    src_dir = r'C:\Users\86183\Downloads\Wi-Pose\Wi-Pose\Train'
    dst_dir = r'C:\Users\86183\Downloads\Wi-Pose\Wi-Pose\validate'

    # 确保源目录存在
    if not os.path.exists(src_dir):
        print(f"源目录 {src_dir} 不存在。")
        exit()

    # 如果目标目录不存在，则创建它
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print(f"目标目录 {dst_dir} 已创建。")

    # 获取源目录下所有的 .mat 文件
    mat_files = [f for f in os.listdir(src_dir) if f.endswith('.mat')]

    # 检查是否有足够多的 .mat 文件
    if len(mat_files) < 1000:
        print(f"源目录中 .mat 文件的数量少于1000个。")
        exit()

    # 随机选择1000个文件
    selected_files = random.sample(mat_files, 1000)

    # 复制文件
    for file_name in selected_files:
        src_file_path = os.path.join(src_dir, file_name)
        dst_file_path = os.path.join(dst_dir, file_name)
        shutil.copy(src_file_path, dst_file_path)
        print(f"文件 {file_name} 已复制到 {dst_dir}")

    print("复制完成。")

def OppMatProcess(mat_file):
   
    #mat_file = 'newtrain/test1.mat'

    # 加载.mat文件
    mat_data = loadmat(mat_file)

    # 提取Mconv7_stage6_L1和Mconv7_stage6_L2，并将它们转换为NumPy数组
    Mconv7_stage6_L1 = mat_data['paf_avg']
    Mconv7_stage6_L2 = mat_data['heatmap_avg']

    # 打印NumPy数组的形状以验证
    #print("Output shape (heatmap): " + str(Mconv7_stage6_L2.shape))
    #print("Output shape (paf): " + str(Mconv7_stage6_L1.shape))

    # 将NumPy数组转换为PyTorch张量
    Mconv7_stage6_L1_tensor = torch.from_numpy(Mconv7_stage6_L1).unsqueeze(0)  # 添加一个批次维度
    Mconv7_stage6_L2_tensor = torch.from_numpy(Mconv7_stage6_L2) .unsqueeze(0) # 添加一个批次维度
    # 使用permute重新排列维度
    Mconv7_stage6_L1_tensor = Mconv7_stage6_L1_tensor.permute(0, 3, 1, 2)
    Mconv7_stage6_L2_tensor = Mconv7_stage6_L2_tensor.permute(0, 3, 1, 2)
    # 打印张量的维度
    #print("Mconv7_stage6_L1_tensor shape:", Mconv7_stage6_L1_tensor.shape)
    #print("Mconv7_stage6_L2_tensor shape:", Mconv7_stage6_L2_tensor.shape)

    # 使用F.interpolate调整最后两个维度到36x36
    Mconv7_stage6_L1_resized = F.interpolate(Mconv7_stage6_L1_tensor, size=(36, 36), mode='bilinear', align_corners=False)
    Mconv7_stage6_L2_resized = F.interpolate(Mconv7_stage6_L2_tensor, size=(36, 36), mode='bilinear', align_corners=False)
     # 归一化每个36x36层的最大值为1
    # 归一化每个36x36层的最大值为1
    #max_values_L1 = Mconv7_stage6_L1_resized.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    #max_values_L2 = Mconv7_stage6_L2_resized.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    #Mconv7_stage6_L1_resized = Mconv7_stage6_L1_resized / max_values_L1
    #Mconv7_stage6_L2_resized = Mconv7_stage6_L2_resized / max_values_L2

    # 打印调整后的张量的形状以验证
    #print("Output shape (heatmap) after resize: " + str(Mconv7_stage6_L2_resized.shape))
    #print("Output shape (paf) after resize: " + str(Mconv7_stage6_L1_resized.shape))

    return Mconv7_stage6_L1_resized,Mconv7_stage6_L2_resized

def OppMatProcess_ST1(mat_file):
   
    mat_data = loadmat(mat_file)
    Mconv7_stage6_L1 = mat_data['paf_avg1']
    Mconv7_stage6_L2 = mat_data['heatmap_avg1']

    # 将NumPy数组转换为PyTorch张量
    Mconv7_stage6_L1_tensor = torch.from_numpy(Mconv7_stage6_L1).unsqueeze(0)  # 添加一个批次维度
    Mconv7_stage6_L2_tensor = torch.from_numpy(Mconv7_stage6_L2) .unsqueeze(0) # 添加一个批次维度
    # 使用permute重新排列维度
    Mconv7_stage6_L1_tensor = Mconv7_stage6_L1_tensor.permute(0, 3, 1, 2)
    Mconv7_stage6_L2_tensor = Mconv7_stage6_L2_tensor.permute(0, 3, 1, 2)[:,:6,:,:]

    # 使用F.interpolate调整最后两个维度到36x36
    Mconv7_stage6_L1_resized = F.interpolate(Mconv7_stage6_L1_tensor, size=(36, 36), mode='bilinear', align_corners=False)
    Mconv7_stage6_L2_resized = F.interpolate(Mconv7_stage6_L2_tensor, size=(36, 36), mode='bilinear', align_corners=False)

    return Mconv7_stage6_L1_resized,Mconv7_stage6_L2_resized

def OppMatProcess_ST2(mat_file):
   
    mat_data = loadmat(mat_file)


    Mconv7_stage6_L1 = mat_data['paf_avg2']
    Mconv7_stage6_L2 = mat_data['heatmap_avg2']

    # 将NumPy数组转换为PyTorch张量
    Mconv7_stage6_L1_tensor = torch.from_numpy(Mconv7_stage6_L1).unsqueeze(0)  # 添加一个批次维度
    Mconv7_stage6_L2_tensor = torch.from_numpy(Mconv7_stage6_L2) .unsqueeze(0) # 添加一个批次维度
    # 使用permute重新排列维度
    Mconv7_stage6_L1_tensor = Mconv7_stage6_L1_tensor.permute(0, 3, 1, 2)
    Mconv7_stage6_L2_tensor = Mconv7_stage6_L2_tensor.permute(0, 3, 1, 2)[:,:6,:,:]

    # 使用F.interpolate调整最后两个维度到36x36
    Mconv7_stage6_L1_resized = F.interpolate(Mconv7_stage6_L1_tensor, size=(36, 36), mode='bilinear', align_corners=False)
    Mconv7_stage6_L2_resized = F.interpolate(Mconv7_stage6_L2_tensor, size=(36, 36), mode='bilinear', align_corners=False)

    return Mconv7_stage6_L1_resized,Mconv7_stage6_L2_resized
def OppMatProcess_ST3(mat_file):
   
    mat_data = loadmat(mat_file)


    Mconv7_stage6_L1 = mat_data['paf_avg3']
    Mconv7_stage6_L2 = mat_data['heatmap_avg3']

    # 将NumPy数组转换为PyTorch张量
    Mconv7_stage6_L1_tensor = torch.from_numpy(Mconv7_stage6_L1).unsqueeze(0)  # 添加一个批次维度
    Mconv7_stage6_L2_tensor = torch.from_numpy(Mconv7_stage6_L2) .unsqueeze(0) # 添加一个批次维度
    # 使用permute重新排列维度
    Mconv7_stage6_L1_tensor = Mconv7_stage6_L1_tensor.permute(0, 3, 1, 2)
    Mconv7_stage6_L2_tensor = Mconv7_stage6_L2_tensor.permute(0, 3, 1, 2)

    # 使用F.interpolate调整最后两个维度到36x36
    Mconv7_stage6_L1_resized = F.interpolate(Mconv7_stage6_L1_tensor, size=(36, 36), mode='bilinear', align_corners=False)
    Mconv7_stage6_L2_resized = F.interpolate(Mconv7_stage6_L2_tensor, size=(36, 36), mode='bilinear', align_corners=False)

    return Mconv7_stage6_L1_resized,Mconv7_stage6_L2_resized

def readOppGT(mat_file):
    # 加载.mat文件
    mat_data = loadmat(mat_file)

    # 提取Mconv7_stage6_L1和Mconv7_stage6_L2，并将它们转换为NumPy数组
    coords = mat_data['coords']
    #print(coords)
    # 初始化列表存储真实的x和y坐标
    true_x = []
    true_y = []

    # 遍历coords数组提取坐标
    for coord in coords:
        x, y, *_ = coord  # 提取x和y坐标
        true_x.append(x)
        true_y.append(y)

    # 打印结果
    #print('true_x:', true_x)
    #print('true_y:', true_y)
    return true_x, true_y

def readOppGTWM(mat_file):
    # 加载.mat文件
    mat_data = loadmat(mat_file)

    # 提取Mconv7_stage6_L1和Mconv7_stage6_L2，并将它们转换为NumPy数组
    coords = mat_data['coords']
    #print(coords)
    # 初始化列表存储真实的x和y坐标
    true_x = []
    true_y = []

    # 遍历coords数组提取坐标
    for coord in coords:
        x, y, *_ = coord  # 提取x和y坐标
        true_x.append(x)
        true_y.append(y)

    # 打印结果
    #print('true_x:', true_x)
    #print('true_y:', true_y)
    return np.array(true_x+true_y, dtype=np.float32)


def DftMatrix(csi_sequence):
    # 步骤2和3: 使用75个样本的窗口滑动并计算18点DFT
    dft_results = []
    window_size = 78
    step_size = 1  # 每次滑动1个样本
    dft_points = 36

    # 计算可以滑动的次数
    num_windows = (len(csi_sequence) - window_size) // step_size 

    for start in range(0, num_windows * step_size, step_size):
        window = csi_sequence[start:start + window_size]  # 获取当前窗口的样本
        hamming_window = np.hamming(window_size)
        window = window.flatten() * hamming_window  # 施加汉明窗
        dft = np.fft.fft(window.T, dft_points)  # 计算18点DFT
        # 应用汉明窗
        
       #print('dftshape',dft.shape)

        dft_results.append(dft)  # 添加DFT结果到列表

    # 步骤4: 将所有DFT结果存储在一个18x75的矩阵中
    dft_matrix = np.array(dft_results).T  # 转置以匹配18x75的形状
    dft_matrix = np.squeeze(dft_matrix)  # 移除长度为1的维度

    return dft_matrix
def CSIInput(mat_file):
    mat_data = loadmat(mat_file)
     # 打印出加载到的NumPy数组的形状
    for key in mat_data:
        if isinstance(mat_data[key], np.ndarray):
            print(f"{key} shape: {mat_data[key].shape}")
    selected_rows = mat_data['selected_rows']
    # 重新塑形数组为(150, 3, 30)
    reshaped_array = selected_rows.reshape(150, 3, 30)
    # 选择整个30维
    random_1x1x30 = reshaped_array[:, 0, :]
    random_1x1x301 = reshaped_array[:, 1, :]
    random_1x1x302 = reshaped_array[:, 2, :]
    # 步骤1: 提取150x1的复数CSI序列
    csi_sequence = random_1x1x30[:, 5].reshape(-1, 1)  # 确保它是150x1的
    csi_sequence1 = random_1x1x301[:, 5].reshape(-1, 1)  # 确保它是150x1的
    csi_sequence2 = random_1x1x302[:, 5].reshape(-1, 1)  # 确保它是150x1的
    print(csi_sequence.shape)
    dft_matrix = DftMatrix(csi_sequence)
    dft_matrix1 = DftMatrix(csi_sequence1)
    dft_matrix2 = DftMatrix(csi_sequence2)
    # 两两相除
    result1 = dft_matrix / dft_matrix1  # dft_matrix除以dft_matrix1
    result2 = dft_matrix / dft_matrix2  # dft_matrix除以dft_matrix2
    result3 = dft_matrix1 / dft_matrix2  # dft_matrix1除以dft_matrix2

        # 计算绝对值和相位
    abs_dft_matrix = np.abs(dft_matrix)
    angle_result1 = np.angle(result1)

    # 归一化abs_dft_matrix
    min_val_abs = np.min(abs_dft_matrix)
    max_val_abs = np.max(abs_dft_matrix)
    normalized_abs_dft_matrix = (abs_dft_matrix - min_val_abs) / (max_val_abs - min_val_abs)

    # 归一化angle_result1
    min_val_angle = np.min(angle_result1)
    max_val_angle = np.max(angle_result1)
    normalized_angle_result1 = (angle_result1 - min_val_angle) / (max_val_angle - min_val_angle)

# 将两个归一化的矩阵沿着第二个轴（列）连接
    combined_matrix1 = np.concatenate((normalized_abs_dft_matrix, normalized_angle_result1), axis=1)
    combined_matrix2 = np.concatenate((abs(dft_matrix1),np.angle(result2) ), axis=1)
    combined_matrix3 = np.concatenate((abs(dft_matrix2),np.angle(result3) ), axis=1)

    final_matrix = np.concatenate((combined_matrix1,combined_matrix2,combined_matrix3 ), axis=0)



    
    # 定义原始和目标的x轴点
 
    #dft_matrix_upsampled = zoom(dft_matrix, (4, 2), order=1)

    
    # 使用seaborn绘制热力图
    plt.figure(figsize=(10, 6))  # 设置图像大小
    sns.heatmap(np.angle(dft_matrix), annot=False, cmap='coolwarm')  # annot=False表示不显示数值，cmap='coolwarm'是颜色映射
    #sns.heatmap(abs(dft_matrix), annot=False, cmap='coolwarm')  # annot=False表示不显示数值，cmap='coolwarm'是颜色映射

    # 设置图像标题和坐标轴标签
    plt.title('Heatmap of DFT Matrix')  # 设置图像标题
    plt.xlabel('Window Index')  # 设置x轴标签
    plt.ylabel('DFT Bin Index')  # 设置y轴标签

    # 显示图像
    plt.show()

    # 打印结果矩阵的形状以确认
    print("DFT matrix shape:", dft_matrix.shape)
    # 随机选择一行
   # 绘制图像
    plt.figure(figsize=(15, 8))  # 设置图像大小
    plt.plot(abs(random_1x1x30))  # 绘制150个样本的值

    # 设置图像标题和坐标轴标签
    plt.title('Subcarrier Index vs Sample Index')  # 设置图像标题
    plt.xlabel('Sample Index')  # 设置x轴标签
    plt.ylabel('Subcarrier Index')  # 设置y轴标签

    # 减少x轴的刻度数量
    plt.xticks(range(0, 150, 10))  # 每10个样本显示一个刻度

    # 减少y轴的刻度数量
    plt.yticks(range(0, 30, 5))  # 每5个子载波显示一个刻度

    # 显示网格
    plt.grid(True)

    # 显示图像
    plt.show()
    #print(mat_data[0])
    # 随机选择150行的索引
    #random_indices = np.random.choice(mat_data.shape[0], 150, replace=False)

    # 使用这些索引来选择150行
    #selected_rows = mat_data[random_indices, :]

    # 保存选中的行到一个新的.mat文件
    #savemat('selected_rows.mat', {'selected_rows': selected_rows})
def complex_upsample(signal, original_length, upsampled_length):
    # 计算上采样倍数 L (可能非整数)
    L = upsampled_length / original_length
    
    # 使用多相滤波器实现分数倍插值（自动处理插零和滤波）
    upsampled = resample_poly(signal, up=upsampled_length, down=original_length, window=('kaiser', 5.0))
    
    return upsampled
def complex_upsample_axis(arr, up_length, axis=0):
    
 

    original_length = arr.shape[axis]
    
    # 定义插值函数
    def interp_1d(signal):
        original_time = np.linspace(0, 1, original_length)
        interp_time = np.linspace(0, 1, up_length)
        real_part = np.interp(interp_time, original_time, signal.real)
        imag_part = np.interp(interp_time, original_time, signal.imag)
        return real_part + 1j * imag_part
    
    return np.apply_along_axis(interp_1d, axis=axis, arr=arr)
def TerminalCSIInput2(mat_file):
    
    mat_data = loadmat(mat_file)
    
    return np.abs(mat_data['MAGNITUDE'])  # 返回绝对值（与原函数逻辑一致）
def TerminalCSIInput(mat_file):
    
    mat_data = loadmat(mat_file)
    
    if 'MAGNITUDE' in mat_data:
        #print(f"Key 'MAGNITUDE' already exists in {mat_file}, returning existing value.")
        return np.abs(mat_data['MAGNITUDE'])  # 返回绝对值（与原函数逻辑一致）'''
     # 打印出加载到的NumPy数组的形状'''
    
    selected_rows = mat_data['csi_data'] 
    reshaped_array = selected_rows[-10:,:].reshape(10, 3, 30)   
    at1 = reshaped_array[:, 0, :]
    at2 = reshaped_array[:, 1, :]
    at3 = reshaped_array[:, 2, :]

    upsampled_at1 = complex_upsample_axis(at1, up_length=128, axis=1)  # 结果 (128,30)
    upsampled_at2 = complex_upsample_axis(at2, up_length=128, axis=1)
    upsampled_at3 = complex_upsample_axis(at3, up_length=128, axis=1)
    
    print(upsampled_at1.shape)

    upsampled_at1 = complex_upsample_axis(abs(upsampled_at1), up_length=128, axis=0)  # 结果 (128,30)
    upsampled_at2 = complex_upsample_axis(abs(upsampled_at2), up_length=128, axis=0)
    upsampled_at3 = complex_upsample_axis(abs(upsampled_at3), up_length=128, axis=0)

    final_array = np.stack((upsampled_at1, upsampled_at2, upsampled_at3), axis=0)
    # 将 finalarray 添加到原 mat_data 中
    mat_data['MAGNITUDE'] = final_array
    savemat(mat_file, mat_data)
    print('成功写入',final_array.shape)
    return abs(final_array)


def PracticalCSIInput(mat_file):
    mat_data = loadmat(mat_file)
     # 打印出加载到的NumPy数组的形状
    
    selected_rows = mat_data['csi_data']
    # 重新塑形数组为(150, 3, 30)
    reshaped_array = selected_rows[-10:,:].reshape(10, 3, 30)
    at1 = reshaped_array[:, 0, :]
    at2 = reshaped_array[:, 1, :]
    at3 = reshaped_array[:, 2, :]
    # 确保数组是 OpenCV 支持的数据类型
    at1 = abs(at1).astype(np.float32)
    at2 = abs(at2).astype(np.float32)
    at3 = abs(at3).astype(np.float32)
    # 上采样插值到 30x30
    upsampled_at1 = cv2.resize(abs(at1), (30, 30), interpolation=cv2.INTER_LINEAR)
    upsampled_at2 = cv2.resize(abs(at2), (30, 30), interpolation=cv2.INTER_LINEAR)
    upsampled_at3 = cv2.resize(abs(at3), (30, 30), interpolation=cv2.INTER_LINEAR)

    # 连接数组
    final_array = np.stack((upsampled_at1, upsampled_at2, upsampled_at3), axis=0)
    
 

    #print(final_array.shape)
    return final_array

       
def FulAttentionCSIinout(mat_file):
    mat_data = loadmat(mat_file)
     # 打印出加载到的NumPy数组的形状
    
    selected_rows = mat_data['csi_data']
    # 重新塑形数组为(150, 3, 30)
    reshaped_array = selected_rows.reshape(10, 3, 30)
    at1 = reshaped_array[:, 0, :]
    at2 = reshaped_array[:, 1, :]
    at3 = reshaped_array[:, 2, :]
    # 确保数组是 OpenCV 支持的数据类型
    at1 = abs(at1).astype(np.float32)
    at2 = abs(at2).astype(np.float32)
    at3 = abs(at3).astype(np.float32)
    # 上采样插值到 30x30
    upsampled_at1 = cv2.resize(abs(at1), (30, 30), interpolation=cv2.INTER_LINEAR)
    upsampled_at2 = cv2.resize(abs(at2), (30, 30), interpolation=cv2.INTER_LINEAR)
    upsampled_at3 = cv2.resize(abs(at3), (30, 30), interpolation=cv2.INTER_LINEAR)

    # 连接数组
    final_array = np.stack((upsampled_at1, upsampled_at2, upsampled_at3), axis=0)
    
 

    #print(final_array.shape)
    return final_array
def PcmReconstruct(jhm1,jhm2,jhm3):
    concatenated_tensor = torch.cat((jhm1, jhm2, jhm3), dim=1)
    #concatenated_tensor = concatenated_tensor.permute(0,1,2,5,8,11,3,6,9,12,14,15,4,7,10,13,16,17,18)
    #print(concatenated_tensor.shape)
    # 定义新的维度顺序
    new_order = torch.tensor([0, 1, 2, 6, 12, 3, 7, 13, 4, 8, 14, 5, 9, 15, 10, 11, 16, 17, 18])

# 使用索引重新排序第二个维度
    reordered_x = concatenated_tensor[:, new_order, :, :]
    #print(reordered_x.shape)
    return  reordered_x 
def PhaseExtract(mat_file):
    mat_data = loadmat(mat_file)
    selected_rows = mat_data['csi_data']
    
    # 重新塑形数组为(10, 3, 30)
    reshaped_array = selected_rows[-10:, :]
    reshaped_array = reshaped_array.reshape(10, 3, 30)
    
    # 提取每个天线的复数CSI数据
    at1 = reshaped_array[:, 0, :]
    at2 = reshaped_array[:, 1, :]
    at3 = reshaped_array[:, 2, :]
    
    # 计算每个天线的相位
    phase_at1 = np.angle(at1)
    phase_at2 = np.angle(at2)
    phase_at3 = np.angle(at3)
    
    # 将相位转换为度数（可选）
    phase_at1_deg = np.degrees(phase_at1)
    phase_at2_deg = np.degrees(phase_at2)
    phase_at3_deg = np.degrees(phase_at3)
    
    #visualize_phase_array(phase_at1_deg[5])
    #isualize_phase_array(adjust_phases(phase_at1_deg[5]))
    
    # 校准相位
    calibrated_phase_at1 = calibrate_phase(phase_at1_deg)
    calibrated_phase_at2 = calibrate_phase(phase_at2_deg)
    calibrated_phase_at3 = calibrate_phase(phase_at3_deg)
    #print(calibrated_phase_at1 [9])
    '''
    visualize_phase_array(calibrated_phase_at1[3])
    visualize_phase_array(calibrated_phase_at1[5])
    visualize_phase_array(calibrated_phase_at1[7])
    visualize_phase_array(calibrated_phase_at1[9])
        
    '''
    upsampled_at1 = cv2.resize(calibrated_phase_at1, (30, 30), interpolation=cv2.INTER_LINEAR)
    upsampled_at2 = cv2.resize(calibrated_phase_at2, (30, 30), interpolation=cv2.INTER_LINEAR)
    upsampled_at3 = cv2.resize(calibrated_phase_at3, (30, 30), interpolation=cv2.INTER_LINEAR)

    # 连接数组
    final_array = np.stack((upsampled_at1, upsampled_at2, upsampled_at3), axis=0)
    # 将结果写回原mat文件
    mat_data['phase'] = final_array
    savemat(mat_file, mat_data)

    return final_array
def PhaseExtract_dir(mat_file):
    mat_data = loadmat(mat_file)
    selected_rows = mat_data['phase']
    
    return  selected_rows
def adjust_phases(phase_array):
    # 确保输入是一个1x30的数组
    if phase_array.shape != (30,):
        raise ValueError("Input array must be of shape (30,)")

    # 初始化调整后的相位数组
    adjusted_phases = phase_array.copy()

    # 从第二个元素开始，确保每个相位值都比前一个相位值小
    for i in range(1, len(adjusted_phases)):
        if adjusted_phases[i] > adjusted_phases[i - 1]+100:
            # 生成一个前i个都是360，后面都是0的数组
            adjustment = np.array([360] * i + [0] * (len(adjusted_phases) - i))
            adjusted_phases += adjustment

    return adjusted_phases
def visualize_phase_array(phase_array):
    # 确保输入是一个1x30的数组
    if phase_array.shape != (30,):
        raise ValueError("Input array must be of shape (30,)")

    # 子载波索引
    subcarrier_indices = np.arange(30)

    # 创建图形
    plt.figure(figsize=(10, 6))

    # 绘制相位
    plt.plot(subcarrier_indices, phase_array, 'bo-', label='Phase')
    plt.title('Phase of Subcarriers')
    plt.xlabel('Subcarrier Index')
    plt.ylabel('Phase (degrees)')
    plt.legend()
    plt.grid(True)

    # 打印相位数组
    print("Phase array:")
    print(phase_array)

    # 显示图形
    plt.show()
def PracticalRemoval(mat_file):
    try:
        mat_data = loadmat(mat_file)  # 加载.mat文件
        if 'csi_data' in mat_data:  # 检查是否存在'csi_data'
            selected_rows = mat_data['csi_data']
            if selected_rows.size == 0:  # 如果csi_data为空
                print(f"Deleting empty CSI file: {mat_file}")
                os.remove(mat_file)  # 删除文件

        else:
            print(f"Warning: No 'csi_data' found in {mat_file}. Deleting file.")
            os.remove(mat_file)  # 如果没有'csi_data'，也删除文件
    except Exception as e:
        print(f"Error processing {mat_file}: {e}")        
def calibrate_phase(phase_data):
    num_time_steps, num_subcarriers = phase_data.shape
    calibrated_phase = np.zeros_like(phase_data)
    
    for t in range(num_time_steps):
        # 计算相位斜率 k 和偏移 b
        phase_data[t]=adjust_phases(phase_data[t])
        k = (phase_data[t, -1] - phase_data[t, 0]) / (num_subcarriers - 1)
        b = np.mean(phase_data[t, :])
        
        # 应用线性变换校准相位
        for i in range(num_subcarriers):
            calibrated_phase[t, i] = phase_data[t, i] -(k * i + b)
    
    return calibrated_phase

if __name__=="__main__":
    #mats = glob.glob('C:/Users\86183\Desktop/result2/result2/*.mat')
    #mats = mats[0]
    #print (mats)
    #delete_unnormal(mats)
    #mat_file = 'newtrain/3.mat'
    #mat_file = 'test.mat'
    #TerminalCSIInput(mat_file)
    #PhaseExtract(mat_file)
    mats = []
    mats += glob.glob('/home/qyy/notebooks/group2_30_percent_bend/*.mat')
    mats += glob.glob('/home/qyy/notebooks/group2_30_percent_crouch/*.mat')
    mats += glob.glob('/home/qyy/notebooks/group2_30_percent_lean/*.mat')
    mats += glob.glob('/home/qyy/notebooks/group2_30_percent_push/*.mat')
    mats += glob.glob('/home/qyy/notebooks/group2_30_percent_sit/*.mat')
    mats += glob.glob('/home/qyy/notebooks/group2_30_percent_stand/*.mat')
    mats += glob.glob('/home/qyy/notebooks/group2_30_percent_walk/*.mat')
    mats += glob.glob('/home/qyy/notebooks/group2_30_percent_wave/*.mat')    
    #mats += glob.glob('/home/qyy/notebooks/group1_70_percent_bend/*.mat')
    #mats += glob.glob('/home/qyy/notebooks/group1_70_percent_crouch/*.mat')
    #mats += glob.glob('/home/qyy/notebooks/group1_70_percent_lean/*.mat')
    #mats += glob.glob('/home/qyy/notebooks/group1_70_percent_push/*.mat')
    #mats += glob.glob('/home/qyy/notebooks/group1_70_percent_sit/*.mat')
    #mats += glob.glob('/home/qyy/notebooks/group1_70_percent_stand/*.mat')
    #mats += glob.glob('/home/qyy/notebooks/group1_70_percent_walk/*.mat')
    #mats += glob.glob('/home/qyy/notebooks/group1_70_percent_wave/*.mat')
    #print(mats)
    #print(mats)
    #mats = mats[:1]
    for index, fl in enumerate(mats, start=1):  # 从 1 开始计数
        #TerminalCSIInput(fl)
        PracticalRemoval(fl)
        print(index)  # 打印当前处理的序号（1, 2, 3...）
    '''

    mats += glob.glob('/mnt/workspace/group1_70_percent_bend/*.mat')
    mats += glob.glob('/mnt/workspace/group1_70_percent_crouch/*.mat')
    mats += glob.glob('/mnt/workspace/group1_70_percent_lean/*.mat')
    mats += glob.glob('/mnt/workspace/group1_70_percent_push/*.mat')
    mats += glob.glob('/mnt/workspace/group1_70_percent_sit/*.mat')
    mats += glob.glob('/mnt/workspace/group1_70_percent_stand/*.mat')
    mats += glob.glob('/mnt/workspace/group1_70_percent_walk/*.mat')
    mats += glob.glob('/mnt/workspace/group1_70_percent_wave/*.mat')
    
    mats += glob.glob('/mnt/workspace/group2_30_percent_bend/*.mat')
    mats += glob.glob('/mnt/workspace/group2_30_percent_crouch/*.mat')
    mats += glob.glob('/mnt/workspace/group2_30_percent_lean/*.mat')
    mats += glob.glob('/mnt/workspace/group2_30_percent_push/*.mat')
    mats += glob.glob('/mnt/workspace/group2_30_percent_sit/*.mat')
    mats += glob.glob('/mnt/workspace/group2_30_percent_stand/*.mat')
    mats += glob.glob('/mnt/workspace/group2_30_percent_walk/*.mat')
    mats += glob.glob('/mnt/workspace/group2_30_percent_wave/*.mat')
    for matf in mats:
        PhaseExtract(matf)
        
        #print(PhaseExtract_dir(mat_file))
    '''
    
    # 使用loadmat函数加载.mat文件
    #mat_data = loadmat(mat_file)

    # 打印出'heatmap_avg'键对应的值
    #print(mat_data['coords'])
    #OppMatProcess(mat_file)
    #readOppGT(mat_file)
    #CSIInput(mat_file)
    #PracticalCSIInput(mats)'''
    '''
    paf1,jhm1 = OppMatProcess_ST1(mat_file)
    paf2,jhm2 = OppMatProcess_ST2(mat_file)
    paf3,jhm3 = OppMatProcess_ST3(mat_file)
    print = (PcmReconstruct(jhm1,jhm2,jhm3).shape)'''


    



