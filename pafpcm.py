import torch
print(torch.__version__)
print(torch.version.cuda)

import numpy as np
import cv2
from mmpose.codecs import AssociativeEmbedding


# 1. 输入数据 (17×2关键点坐标)
keypoints = np.random.rand(17, 2) * 100  # 模拟数据，实际需替换为真实坐标
keypoints = keypoints[np.newaxis, :, :]  # 扩展为 (1, 17, 2)

# 2. 初始化编码器
encoder = AssociativeEmbedding(
    input_size=(640, 480),
    heatmap_size=(36, 36),
    sigma=4.0
)

# 3. 生成PCM和PAF
encoded = encoder.encode(keypoints)
print(encoded.keys())  # 检查输出是否包含 'pafs' 或类似字段
heatmaps = encoded['heatmaps']  # (1, 17, H, W)
print(heatmaps.shape)
#pafs = encoded['pafs']         # (1, 38, H, W)

# 4. 可视化
cv2.imwrite('heatmap.jpg', cv2.normalize(heatmaps[0, 0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
#cv2.imwrite('paf.jpg', cv2.normalize(pafs[0, :2], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))