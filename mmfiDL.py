import os
import scipy.io as scio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# --------------------- 1. 配置参数 ---------------------
config = {
    "data_unit": "frame",           # 帧模式（非序列模式）
    "modality": "wifi-csi|rgb",     # 指定CSI和RGB模态（可选：infra1/infra2/depth）
    "protocol": "all",              # 使用所有动作
    "split_to_use": "random_split", # 数据划分策略
    "random_split": {
        "random_seed": 42,          # 随机种子
        "ratio": 0.7                # 训练集比例
    }
}

# --------------------- 2. 数据集类定义 ---------------------
class MMFi_Database:
    def __init__(self, data_root):
        self.data_root = data_root
        self.scenes = {}
        self.subjects = {}
        self.actions = {}
        self.modalities = {}
        self.load_database()

    def load_database(self):
        for scene in sorted(os.listdir(self.data_root)):
            if scene.startswith("."):
                continue
            self.scenes[scene] = {}
            for subject in sorted(os.listdir(os.path.join(self.data_root, scene))):
                if subject.startswith("."):
                    continue
                self.scenes[scene][subject] = {}
                self.subjects[subject] = {}
                for action in sorted(os.listdir(os.path.join(self.data_root, scene, subject))):
                    if action.startswith("."):
                        continue
                    self.scenes[scene][subject][action] = {}
                    self.subjects[subject][action] = {}
                    if action not in self.actions.keys():
                        self.actions[action] = {}
                    if scene not in self.actions[action].keys():
                        self.actions[action][scene] = {}
                    if subject not in self.actions[action][scene].keys():
                        self.actions[action][scene][subject] = {}
                    for modality in ['infra1', 'infra2', 'depth', 'rgb', 'lidar', 'mmwave', 'wifi-csi']:
                        data_path = os.path.join(self.data_root, scene, subject, action, modality)
                        self.scenes[scene][subject][action][modality] = data_path
                        self.subjects[subject][action][modality] = data_path
                        self.actions[action][scene][subject][modality] = data_path
                        if modality not in self.modalities.keys():
                            self.modalities[modality] = {}
                        if scene not in self.modalities[modality].keys():
                            self.modalities[modality][scene] = {}
                        if subject not in self.modalities[modality][scene].keys():
                            self.modalities[modality][scene][subject] = {}
                        if action not in self.modalities[modality][scene][subject].keys():
                            self.modalities[modality][scene][subject][action] = data_path

class MMFi_Dataset(Dataset):
    def __init__(self, data_base, data_unit, modality, split, data_form):
        self.data_base = data_base
        self.data_unit = data_unit
        self.modality = modality.split('|')
        for m in self.modality:
            assert m in ['rgb', 'infra1', 'infra2', 'depth', 'lidar', 'mmwave', 'wifi-csi']
        self.split = split
        self.data_source = data_form
        self.data_list = self.load_data()

    def get_scene(self, subject):
        if subject in ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09', 'S10']:
            return 'E01'
        elif subject in ['S11', 'S12', 'S13', 'S14', 'S15', 'S16', 'S17', 'S18', 'S19', 'S20']:
            return 'E02'
        elif subject in ['S21', 'S22', 'S23', 'S24', 'S25', 'S26', 'S27', 'S28', 'S29', 'S30']:
            return 'E03'
        elif subject in ['S31', 'S32', 'S33', 'S34', 'S35', 'S36', 'S37', 'S38', 'S39', 'S40']:
            return 'E04'
        else:
            raise ValueError('Subject does not exist in this dataset.')

    def get_data_type(self, mod):
        if mod in ["rgb", 'infra1', "infra2"]:
            return ".npy"
        elif mod in ["lidar", "mmwave"]:
            return ".bin"
        elif mod in ["depth"]:
            return ".png"
        elif mod in ["wifi-csi"]:
            return ".mat"
        else:
            raise ValueError("Unsupported modality.")

    def load_data(self):
        data_info = []
        for subject, actions in self.data_source.items():
            for action in actions:
                frame_num = 297  # 每个动作固定297帧
                for idx in range(frame_num):
                    data_dict = {
                        'modality': self.modality,
                        'scene': self.get_scene(subject),
                        'subject': subject,
                        'action': action,
                        'gt_path': os.path.join(self.data_base.data_root, self.get_scene(subject), subject, action, 'ground_truth.npy'),
                        'idx': idx
                    }
                    data_valid = True
                    for mod in self.modality:
                        data_dict[f'{mod}_path'] = os.path.join(
                            self.data_base.data_root, self.get_scene(subject), subject, action, mod,
                            f"frame{idx+1:03d}{self.get_data_type(mod)}"
                        )
                        if not os.path.exists(data_dict[f'{mod}_path']):
                            data_valid = False
                    if data_valid:
                        data_info.append(data_dict)
        return data_info

    def read_frame(self, frame):
        _mod, _frame = os.path.split(frame)
        _, mod = os.path.split(_mod)
        if mod in ['infra1', 'infra2', 'rgb']:
            data = np.load(frame)
        elif mod == 'depth':
            data = cv2.imread(frame, cv2.IMREAD_UNCHANGED) * 0.001
        elif mod == 'wifi-csi':
            data = scio.loadmat(frame)['CSIamp']
            data[np.isinf(data)] = np.nan
            for i in range(data.shape[2]):  # 处理每个天线
                temp_col = data[:, :, i]
                temp_col[np.isnan(temp_col)] = np.nanmean(temp_col)
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
        else:
            raise ValueError(f"Unsupported modality: {mod}")
        return data

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        gt = torch.from_numpy(np.load(item['gt_path'])[item['idx']])
        sample = {
            'modality': item['modality'],
            'scene': item['scene'],
            'subject': item['subject'],
            'action': item['action'],
            'idx': item['idx'],
            'output': gt
        }
        for mod in item['modality']:
            sample[f'input_{mod}'] = self.read_frame(item[f'{mod}_path'])
        return sample

# --------------------- 3. 数据加载工具函数 ---------------------
def decode_config(config):
    all_subjects = [f'S{i:02d}' for i in range(1, 41)]
    all_actions = [f'A{i:02d}' for i in range(1, 28)]
    
    if config['protocol'] == 'protocol1':
        actions = ['A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
    elif config['protocol'] == 'protocol2':
        actions = ['A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A15', 'A16', 'A24', 'A25', 'A26']
    else:
        actions = all_actions

    if config['split_to_use'] == 'random_split':
        rs = config['random_split']['random_seed']
        ratio = config['random_split']['ratio']
        np.random.seed(rs)
        idx = np.random.permutation(len(all_subjects))
        idx_train = idx[:int(ratio * len(all_subjects))]
        idx_val = idx[int(ratio * len(all_subjects)):]
        subjects_train = np.array(all_subjects)[idx_train].tolist()
        subjects_val = np.array(all_subjects)[idx_val].tolist()
        train_form = {s: actions for s in subjects_train}
        val_form = {s: actions for s in subjects_val}
    else:
        raise NotImplementedError("仅支持random_split")

    return {
        'train_dataset': {'modality': config['modality'], 'split': 'training', 'data_form': train_form},
        'val_dataset': {'modality': config['modality'], 'split': 'validation', 'data_form': val_form}
    }

def make_dataset(dataset_root, config):
    database = MMFi_Database(dataset_root)
    config_dataset = decode_config(config)
    train_dataset = MMFi_Dataset(database, config['data_unit'], **config_dataset['train_dataset'])
    val_dataset = MMFi_Dataset(database, config['data_unit'], **config_dataset['val_dataset'])
    return train_dataset, val_dataset

def collate_fn_padd(batch):
    batch_data = {
        'modality': batch[0]['modality'],
        'scene': [sample['scene'] for sample in batch],
        'subject': [sample['subject'] for sample in batch],
        'action': [sample['action'] for sample in batch],
        'idx': [sample['idx'] for sample in batch],
        'output': torch.stack([sample['output'] for sample in batch])
    }
    for mod in batch[0]['modality']:
        batch_data[f'input_{mod}'] = torch.stack([torch.FloatTensor(sample[f'input_{mod}']) for sample in batch])
    return batch_data

def make_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_padd,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=4
    )

# --------------------- 4. 主程序 ---------------------
if __name__ == "__main__":
    # 初始化
    dataset_root = "/home/qyy/notebooks/MMFi"  # 替换为实际路径
    torch.manual_seed(42)

    # 加载数据集
    train_dataset, val_dataset = make_dataset(dataset_root, config)
    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    # 检查第一个样本
    sample = train_dataset[0]
    print("\n样本示例:")
    print(f"场景: {sample['scene']}, 受试者: {sample['subject']}, 动作: {sample['action']}, 帧索引: {sample['idx']}")
    print(f"CSI形状: {sample['input_wifi-csi'].shape} (子载波, 时间样本, 天线)")
    print(f"RGB形状: {sample['input_rgb'].shape}")
    print(f"标签: {sample['output']}")

    # 创建数据加载器
    train_loader = make_dataloader(train_dataset, batch_size=8)
    val_loader = make_dataloader(val_dataset, batch_size=8, shuffle=False)

    # 验证批处理
    print("\n第一批训练数据:")
    batch = next(iter(train_loader))
    print(f"批大小: {batch['output'].shape[0]}")
    print(f"CSI批形状: {batch['input_wifi-csi'].shape}")  # (B, 30, 100, 3)
    print(f"RGB批形状: {batch['input_rgb'].shape}")      # (B, H, W, 3)