import numpy as np
import cv2
from typing import Dict
from scipy.io import savemat
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

def generate_pose_maps(npy_path: str, output_height: int = 36, output_width: int = 36) -> Dict[str, np.ndarray]:
    """生成PCM和PAF的核心函数（与原代码一致）"""
    keypoints = np.load(npy_path)
    limb_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9),
        (6, 8), (8, 10), (11, 12), (11, 13), (13, 15), (12, 14),
        (14, 16), (5, 11), (6, 12), (3, 5), (4, 6), (5, 6)
    ]

    def _generate_pcm(kpts, height=480, width=640, sigma=20):
        pcm = np.zeros((17, height, width), dtype=np.float32)
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        for j in range(17):
            x, y = kpts[j]
            if x > 0 and y > 0:
                dist = (xx - x)**2 + (yy - y)**2
                pcm[j] = np.exp(-dist / (2 * sigma**2))
        neck = (kpts[5] + kpts[6]) / 2 if (kpts[5][0] > 0 and kpts[6][0] > 0) else np.array([-1, -1])
        neck_pcm = np.exp(-((xx - neck[0])**2 + (yy - neck[1])**2) / (2 * sigma**2)) if neck[0] > 0 else np.zeros((height, width))
        max_pcm = pcm.max(axis=0)
        return np.concatenate([pcm, neck_pcm[np.newaxis], max_pcm[np.newaxis]], axis=0)

    def _generate_paf(kpts, connections, height=480, width=640, limb_width=25):
        paf = np.zeros((2 * len(connections), height, width), dtype=np.float32)
        for i, (j1, j2) in enumerate(connections):
            pt1, pt2 = kpts[j1], kpts[j2]
            if all(v > 0 for v in [*pt1, *pt2]):
                vec = pt2 - pt1
                norm = np.linalg.norm(vec)
                if norm >= 1e-5:
                    unit_vec = vec / norm
                    mask = np.zeros((height, width), dtype=np.float32)
                    cv2.line(mask, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color=1.0, thickness=limb_width)
                    paf[2*i] = unit_vec[0] * mask
                    paf[2*i + 1] = unit_vec[1] * mask
        return paf

    pcm_full = _generate_pcm(keypoints)
    paf_full = _generate_paf(keypoints, limb_connections)
    pcm_out = np.array([cv2.resize(ch, (output_width, output_height)) for ch in pcm_full])
    paf_out = np.array([cv2.resize(ch, (output_width, output_height)) for ch in paf_full])

    return {
        'pcm': pcm_out.astype(np.float32),
        'paf': paf_out.astype(np.float32),
        'limb_connections': np.array(limb_connections),
        'keypoints': keypoints.astype(np.float32)
    }

_file_lock = threading.Lock()

def save_mat_threadsafe(mat_data: Dict[str, np.ndarray], output_path: str) -> None:
    """线程安全的.mat文件保存"""
    with _file_lock:
        try:
            savemat(output_path, mat_data)
        except Exception as e:
            raise IOError(f"保存失败 {output_path}: {str(e)}")

def process_single_file(input_path: str, output_path: str) -> None:
    """处理单个文件的任务单元"""
    try:
        mat_data = generate_pose_maps(input_path)
        save_mat_threadsafe(mat_data, output_path)
    except Exception as e:
        print(f"处理错误 {input_path}: {str(e)}")
        raise

def process_rgb_npy_files(
    base_input_dir: str = "/home/qyy/notebooks/MMFi/E01",
    base_output_dir: str = "/home/qyy/notebooks/MMFi_label",
    max_workers: int = 8,
    verbose: bool = True
) -> None:
    """
    多线程处理所有rgb子目录下的.npy文件
    修改点：仅处理路径中包含'/rgb/'的.npy文件
    """
    file_paths = []
    for root, _, files in os.walk(base_input_dir):
        # 关键筛选条件：只处理rgb子目录
        if 'rgb' in root.replace('\\', '/'):  # 统一路径分隔符
            for file in files:
                if file.endswith('.npy'):
                    input_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, base_input_dir)
                    output_root = os.path.join(base_output_dir, relative_path)
                    os.makedirs(output_root, exist_ok=True)
                    output_file = file.replace('.npy', '.mat')
                    output_path = os.path.join(output_root, output_file)
                    file_paths.append((input_path, output_path))

    # 多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for input_path, output_path in file_paths:
            futures.append(executor.submit(process_single_file, input_path, output_path))
        
        # 进度条
        if verbose:
            for future in tqdm(as_completed(futures), total=len(futures), desc="处理RGB模态文件"):
                future.result()
        else:
            for future in as_completed(futures):
                future.result()

if __name__ == "__main__":
    # 示例调用：仅处理E01下所有rgb子目录的.npy文件
    process_rgb_npy_files(
        base_input_dir="/home/qyy/notebooks/MMFi/E01",
        base_output_dir="/home/qyy/notebooks/MMFi_label",
        max_workers=16,
        verbose=True
    )