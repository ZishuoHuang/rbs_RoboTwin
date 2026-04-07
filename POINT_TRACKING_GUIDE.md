# 点追踪可视化 - 技术说明

## 概述
这个方案实现了**物体表面点在操作过程中的追踪**，显示机械臂夹取物体时采样点的运动轨迹。

## 核心原理

### 方法：ICP配准追踪物体运动

```
第0帧                后续帧
┌─────────────────┬──────────────────┐
│ 采样300个点 ◉◉◉ │ 物体发生运动 ◉◉◉ │
│ 记录世界坐标    │ 计算变换矩阵     │
│ points_world_0  │ R, t = ICP配准   │
└─────────────────┴──────────────────┘
                     ↓
         应用变换: p_new = R·p_0 + t
                     ↓
              投影回图像平面
```

### 处理流程

1. **第0帧点采样**
   - 找到目标物体（面积最大的actor_id = 165）
   - 从14,406个像素中随机采样300个点
   - 使用深度图逆投影得到世界坐标系3D点

2. **后续帧物体追踪**
   - 对每帧提取目标actor的完整点云
   - 使用ICP (Iterative Closest Point) 配准
   - 计算从第0帧到当前帧的刚体变换 R(旋转) + t(平移)

3. **点变换投影**
   ```
   p_world,cur = R @ p_world,0 + t
   p_cam,cur = T_w2c @ p_world,cur
   p_pixel = K @ p_cam / z
   ```

4. **可视化绘制**
   - 彩色点表示采样的位置（蓝→红渐变）
   - 轨迹线连接相邻帧的对应点
   - 绿色文字显示追踪点数和当前帧号

## 输出文件

```
point_tracking_clean/
├── point_tracking.gif      (1.4 MB)  - 10帧GIF动画，循环播放
├── point_tracking.mp4      (438 KB)  - MP4视频格式
└── frame_*.png             - 各单独帧
    ├── frame_000.png       - 第0帧：采样点初始位置
    ├── frame_038.png       - 夹取过程中
    ├── frame_077.png       - 物体在上升过程
    └── frame_116.png       - 最终放置位置
```

## 关键参数说明

| 参数 | 取值 | 说明 |
|------|------|------|
| `num_keyframes` | 10 | 关键帧数量（帧均匀分布在整个序列） |
| `num_points` | 300 | 采样点数量 |
| `actor_id` | 165 | 多彩瓶子ID（自动选择面积最大的物体） |
| `camera_name` | head_camera | 相机视角 |

## 性能指标

- 动画生成时间：约30秒
- GIF文件大小：1.4 MB
- MP4文件大小：438 KB
- 点云下采样：5000个点（加速ICP计算）

## 优化建议

### 如果轨迹仍然不清晰：

1. **减少轨迹线密度**（已实施）
   ```python
   step = max(1, len(pixels_cur) // 50)  # 最多50条轨迹线
   ```

2. **增加关键帧数**以获得更连贯的动画
   ```bash
   python render_point_tracking_v2.py ... --num-keyframes 15
   ```

3. **使用其他相机视角**
   ```bash
   python render_point_tracking_v2.py ... --camera front_camera
   ```

## 使用方法

```bash
# 基本用法
python render_point_tracking_v2.py <HDF5_PATH>

# 自定义参数
python render_point_tracking_v2.py data/episode0.hdf5 \
  --output my_output \
  --num-keyframes 12 \
  --num-points 400

# 结果查看
open point_tracking_clean/point_tracking.gif    # macOS
xdg-open point_tracking_clean/point_tracking.mp4  # Linux
```

## 技术细节

### ICP配准算法
```
输入：source点云 (第0帧)，target点云 (当前帧)
过程：
  1. 中心化两个点云
  2. SVD分解计算最优旋转矩阵 R
  3. 计算平移向量 t
  4. 迭代直至收敛
输出：变换矩阵 T = [R|t]
```

### 点投影公式
```
世界坐标  →  相机坐标  →  图像坐标
p_w          p_c = T_w2c @ p_w      p_img = K @ p_c / z
              (4×4)                  (3×3)
```

## 常见问题

**Q: 点的轨迹是否精确？**  
A: 精确度取决于：
- ICP配准的收敛质量
- 物体点云的完整性（遮挡会降低精度）
- 相机参数的准确性

**Q: 能否追踪多个物体？**  
A: 可以，修改脚本中的actor_id选择逻辑，为每个物体生成不同的可视化

**Q: 支持哪些相机视角？**  
A: 支持所有HDF5中的相机：head_camera, front_camera, left_camera, right_camera

---

## 后续工作

1. **多物体追踪** - 同时显示多个物体的点运动
2. **3D轨迹可视化** - 在3D视图中显示点云运动
3. **轨迹统计** - 计算各点的运动距离和速度
4. **更精确配准** - 使用深度学习的特征匹配替代ICP
