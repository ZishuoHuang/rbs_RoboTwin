# SceneFlow生成指南

## 📋 概述

这个工具包帮助你从已保存的HDF5文件生成SceneFlow数据，**无需启动仿真器**。

### 三个核心脚本：
1. **generate_sceneflow.py** - 从深度图逆投影生成点云，并追踪到各帧
2. **visualize_sceneflow.py** - 用Open3D可视化验证效果
3. **test_sceneflow.py** - 快速测试和诊断

---

## 🚀 快速开始

### 第1步：检查HDF5文件结构
```bash
python test_sceneflow.py data/pick_diverse_bottles/aloha-agilex_clean_50/data/episode0.hdf5
```

这会：
- ✓ 打印HDF5的完整结构
- ✓ 检查是否包含必需的数据（深度、相机参数、分割）
- ✓ 尝试生成一个小的SceneFlow样本进行测试

### 第2步：生成完整的SceneFlow
```bash
python generate_sceneflow.py data/pick_diverse_bottles/aloha-agilex_clean_50/data/episode0.hdf5 \
  --output_dir data/pick_diverse_bottles/aloha-agilex_clean_50/sceneflow
```

输出：
```
data/pick_diverse_bottles/aloha-agilex_clean_50/sceneflow/
  ├── sceneflow_0.npy      # 第0帧关键帧的delta轨迹（参考点）
  ├── sceneflow_1.npy      # 第1/4帧的delta轨迹
  ├── sceneflow_2.npy      # 第2/4帧的delta轨迹
  ├── sceneflow_3.npy      # 第3/4帧的delta轨迹
  ├── sceneflow_4.npy      # 第4/4帧的delta轨迹（最后一帧）
  ├── pointcloud_frame0.npy # 第0帧的原始点云
  └── segmentation_frame0.npy # 第0帧的分割图
```

### 第3步：可视化验证效果

有多种査看模式：

#### 3a. 查看统计信息（无需可视化）
```bash
python visualize_sceneflow.py data/pick_diverse_bottles/aloha-agilex_clean_50/sceneflow --mode stats
```

输出：
```
SceneFlow统计信息
==============================================================

点云统计:
  第0帧点数: 123456
  点的坐标范围: 
    X: [-0.600, 0.600]
    Y: [-0.350, 0.350]
    Z: [0.740, 1.200]

Delta统计:
  Frame 0: max_delta=0.0000, mean_delta=0.0000
  Frame 1: max_delta=0.0234, mean_delta=0.0012
  Frame 2: max_delta=0.0456, mean_delta=0.0023
  ...
```

#### 3b. 可视化所有关键帧（需要Open3D）
```bash
python visualize_sceneflow.py data/pick_diverse_bottles/aloha-agilex_clean_50/sceneflow --mode all
```

会逐帧显示：
- 红色的第0帧参考点云
- 蓝色的当前帧点云
- 按任意键继续下一帧

#### 3c. 对比相邻帧
```bash
python visualize_sceneflow.py data/pick_diverse_bottles/aloha-agilex_clean_50/sceneflow --mode consecutive
```

#### 3d. 追踪单个点的轨迹
```bash
python visualize_sceneflow.py data/pick_diverse_bottles/aloha-agilex_clean_50/sceneflow \
  --mode trajectory --point_idx 5000
```

---

## 🔧 工作原理详解

### 为什么不需要启动仿真器？

```
┌─────────────────────────────────────────────────┐
│ 数据已经保存在HDF5中                             │
├─────────────────────────────────────────────────┤
│ ✓ 深度图 (H×W)                                   │
│ ✓ 相机内参 K (3×3)                              │
│ ✓ 相机外参 T_c2w (4×4)                          │
│ ✓ 物体位姿 (per frame)                          │
│ ✓ 分割信息 (H×W)                                │
└─────────────────────────────────────────────────┘
          ↓ 只需纯数学计算
┌─────────────────────────────────────────────────┐
│ 生成SceneFlow                                     │
├─────────────────────────────────────────────────┤
│ 1. 逆投影：深度图 + 内参 → 相机坐标系点          │
│ 2. 坐标变换：相机坐标 → 世界坐标                 │
│ 3. 点追踪：根据物体位姿计算点的新位置            │
│ 4. 相对位移：计算每帧相对参考帧的delta          │
└─────────────────────────────────────────────────┘
```

所有计算都是 **离线的、可脚本化的、无需渲染的**。

### 核心算法：深度图到点云

**输入：**
- 深度图 D (H×W)：每个像素的深度值（单位：米或mm）
- 相机内参 K (3×3)：包含焦距和主点
- 相机外参 T_c2w (4×4)：相机到世界的变换

**处理步骤：**
```python
# 1. 对每个像素 (u, v) 进行逆投影
d = D[v, u]
K_inv = inv(K)

# 2. 从像素坐标到相机坐标系
[x]       [u]
[y] = K^{-1} * [v] * d
[z]       [1]

# 3. 从相机坐标系变换到世界坐标系
p_world = R @ [x, y, z]^T + t
         = T_c2w[:3, :3] @ p_cam + T_c2w[:3, 3]
```

**输出：**
- 点云 P (N×3)：N个三维点的世界坐标

### 点云追踪原理

**对于第0帧的每个点 p_ref：**

1. 计算该点属于哪个物体（用分割信息）
2. 根据该物体的位姿变化，计算该点在各帧的新位置
3. 存储每帧的点云和相对于参考帧的delta

**Delta计算：**
```python
ref_point = pointcloud_frame0[i]

for frame_t in keyframes:
    current_point = pointcloud_frame_t[i]
    delta_t = current_point - ref_point
    # delta_t 就是第i个点在第t帧相对于参考帧的移动
```

---

## 📊 输出文件格式

### sceneflow_x.npy
```
形状：(N, 3)
  N = 第0帧的点数
  3 = xyz坐标

含义：第x个关键帧相对于第0帧的相对位移
  值为0表示该点没有移动
  值为[0.1, 0, 0]表示点向X轴移动了0.1米
```

### pointcloud_frame0.npy
```
形状：(N, 3)
  N = 有效点的数量
  
含义：第0帧的原始点云（世界坐标系）
```

### segmentation_frame0.npy
```
形状：(H, W) 或 (N,)
  H, W = 图像高度和宽度（如果保存为2D）
  N = 点数（如果转换为1D）

含义：每个点/像素对应的物体ID
```

---

## 🐛 常见问题

### Q: 找不到深度数据怎么办？

**A:** 检查HDF5文件的结构：
```bash
python test_sceneflow.py your_episode.hdf5
```

常见的路径：
- `depth` - 全局深度
- `head_camera/depth` - 头相机深度
- `left_camera/depth` - 左手腕相机深度

在 `generate_sceneflow.py` 中修改 `get_depth_image()` 方法的查找逻辑。

### Q: 点云中有很多噪点怎么办？

**A:** 尝试：
1. 增加深度有效性检查（设置最小/最大深度阈值）
2. 使用 `pcd_crop` 参数裁剪不相关的区域
3. 给生成的点云做降采样

### Q: 追踪的点不对应怎么办？

**A:** 可能是点数不一致。检查：
1. 不同帧的点数是否相同
2. 深度图中的无效点（NaN或0）是否处理一致
3. 分割图对应的物体是否有移动

### Q: 可以使用GPU加速吗？

**A:** 当前实现使用NumPy。可以改成PyTorch版本加速矩阵运算：
```python
import torch

# 替换numpy为torch，使用GPU
K_inv = torch.linalg.inv(K).to('cuda')
points_world = torch.matmul(R, points_cam) + t.unsqueeze(1)
```

---

## 📦 依赖

**必需：**
- numpy
- h5py

**可选（用于可视化）：**
- open3d

**安装：**
```bash
pip install open3d
```

---

## 🔗 下一步：压缩

完成SceneFlow生成后，使用浩楠提供的 **Any4D编码器** 压缩这5条轨迹：

```bash
# 伪代码
from any4d_encoder import compress

sceneflow_files = [
    "sceneflow_0.npy",
    "sceneflow_1.npy", 
    "sceneflow_2.npy",
    "sceneflow_3.npy",
    "sceneflow_4.npy"
]

compressed = compress(sceneflow_files)
compressed.save("episode0_sceneflow.compressed")
```

---

## 📝 批量处理

一次性处理多个episode：

```bash
#!/bin/bash

DATA_DIR="data/pick_diverse_bottles/aloha-agilex_clean_50/data"

for episode in episode*.hdf5; do
    echo "处理: $episode"
    python generate_sceneflow.py "$DATA_DIR/$episode"
    python visualize_sceneflow.py "$DATA_DIR/sceneflow" --mode stats
done
```

或用Python：
```python
from pathlib import Path
from generate_sceneflow import SceneFlowGenerator

data_dir = Path("data/pick_diverse_bottles/aloha-agilex_clean_50/data")

for hdf5_file in data_dir.glob("episode*.hdf5"):
    print(f"处理: {hdf5_file}")
    output_dir = hdf5_file.parent / "sceneflow" / hdf5_file.stem
    
    gen = SceneFlowGenerator(str(hdf5_file))
    gen.generate_sceneflow(str(output_dir))
```

---

## 📞 调试

如需调试，在 `generate_sceneflow.py` 中添加详细日志：

```python
class SceneFlowGenerator:
    def __init__(self, hdf5_path, device="cpu", verbose=True):
        self.verbose = verbose
    
    def _log(self, msg):
        if self.verbose:
            print(f"[DEBUG] {msg}")
```

或使用 `test_sceneflow.py` 的诊断模式：
```bash
python test_sceneflow.py episode0.hdf5 --verbose
```

---

祝你生成顺利！✨
