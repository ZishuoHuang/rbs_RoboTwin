"""
点追踪可视化脚本 - 显示第0帧采样的点在后续帧中的运动轨迹
功能：
1. 从RGB图像背景显示
2. 在第0帧时从特定物体采样点
3. 追踪这些点的3D运动，投影到RGB平面
4. 绘制轨迹线和运动轨迹
"""

import h5py
import numpy as np
from pathlib import Path
import cv2
from typing import Dict, Tuple, List
import argparse
from PIL import Image, ImageDraw
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from generate_sceneflow import SceneFlowGenerator


class PointTrackingVisualizer:
    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self.gen = SceneFlowGenerator(hdf5_path)
        self.gen.load_hdf5()
        
        # 获取相机参数（第0帧）
        self.K = self.gen.get_camera_intrinsic(0)
        self.T_c2w_0 = self.gen.get_camera_extrinsic(0)
        self.T_w2c_0 = np.linalg.inv(self.T_c2w_0)
        
    def get_rgb(self, frame_idx: int, camera_name: str = "head_camera") -> np.ndarray:
        """获取RGB图像"""
        possible_paths = [
            f'observation/{camera_name}/rgb',
            f'{camera_name}/rgb',
            'rgb',
        ]
        
        for path in possible_paths:
            if path in self.gen.hdf5:
                rgb = self.gen.hdf5[path][frame_idx]
                # 处理不同的数据格式
                if isinstance(rgb, bytes):
                    # 如果是bytes，可能是JPEG编码，需要解码
                    import io
                    rgb = cv2.imdecode(np.frombuffer(rgb, np.uint8), cv2.IMREAD_COLOR)
                    if rgb is None:
                        continue
                    return rgb
                elif rgb.dtype != np.uint8:
                    # 如果是float，转换为uint8
                    if rgb.max() <= 1.0:
                        rgb = (rgb * 255).astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)
                return rgb
        
        # 如果没有RGB，用深度图生成灰度图
        depth = self.gen.get_depth_image(frame_idx, camera_name)
        depth_valid = depth[~np.isnan(depth)]
        if len(depth_valid) == 0:
            depth_normalized = np.zeros_like(depth, dtype=np.uint8)
        else:
            depth_normalized = ((depth - depth_valid.min()) / (depth_valid.max() - depth_valid.min()) * 255).astype(np.uint8)
            depth_normalized[np.isnan(depth)] = 0
        return cv2.cvtColor(depth_normalized, cv2.COLOR_GRAY2RGB)
    
    def sample_foreground_points(self, frame_idx: int = 0, camera_name: str = "head_camera", 
                                num_points: int = 500, target_object_id: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        从第0帧的前景物体采样点
        Args:
            frame_idx: 帧索引
            camera_name: 相机名称
            num_points: 采样点数
            target_object_id: 指定要采样的物体ID，如果为None则选择面积最大的物体
        Returns:
            points_3d: (N, 3) 3D点坐标 (世界坐标系)
            pixel_coords: (N, 2) 对应的像素坐标 (u, v)
        """
        depth = self.gen.get_depth_image(frame_idx, camera_name)
        seg = self.gen.get_segmentation(frame_idx, camera_name)
        
        H, W = depth.shape
        
        # 找到所有前景像素 (seg != 0, depth有效)
        valid_mask = (depth > 0) & (~np.isnan(depth)) & (seg > 0)
        
        if np.sum(valid_mask) == 0:
            print("⚠ 警告: 没有有效的前景像素")
            return np.array([]), np.array([])
        
        # 如果没有指定物体ID，选择面积最大的物体
        if target_object_id is None:
            unique_ids, counts = np.unique(seg[valid_mask], return_counts=True)
            target_object_id = unique_ids[np.argmax(counts)]
            print(f"   选择面积最大的物体: ID={target_object_id} (像素数: {counts.max()})")
        
        # 获取该物体的像素
        object_mask = (seg == target_object_id) & valid_mask
        object_pixels = np.where(object_mask)
        v_coords, u_coords = object_pixels[0], object_pixels[1]
        
        if len(u_coords) == 0:
            print(f"⚠ 警告: 物体ID {target_object_id} 没有有效像素")
            return np.array([]), np.array([])
        
        # 随机采样
        if len(u_coords) > num_points:
            indices = np.random.choice(len(u_coords), num_points, replace=False)
            u_coords = u_coords[indices]
            v_coords = v_coords[indices]
        
        # 获取这些像素的深度值
        depths = depth[v_coords, u_coords]
        
        # 逆投影得到相机坐标系的3D点
        points_cam = np.zeros((len(u_coords), 3))
        points_cam[:, 0] = (u_coords - self.K[0, 2]) * depths / self.K[0, 0]  # x
        points_cam[:, 1] = (v_coords - self.K[1, 2]) * depths / self.K[1, 1]  # y
        points_cam[:, 2] = depths  # z
        
        # 转换到世界坐标系
        points_cam_homo = np.hstack([points_cam, np.ones((len(points_cam), 1))])
        points_world = (self.T_c2w_0 @ points_cam_homo.T).T[:, :3]
        
        return points_world, np.column_stack([u_coords, v_coords])
    
    def project_points_to_frame(self, points_world: np.ndarray, frame_idx: int, 
                               camera_name: str = "head_camera") -> Tuple[np.ndarray, np.ndarray]:
        """
        将世界坐标系的点投影到指定帧的图像平面
        Returns:
            pixel_coords: (N, 2) 像素坐标 [u, v]
            in_view: (N,) 布尔数组，表示点是否在视图内
        """
        # 获取该帧的相机参数
        T_c2w = self.gen.get_camera_extrinsic(frame_idx, camera_name)
        T_w2c = np.linalg.inv(T_c2w)
        
        # 转换到该帧的相机坐标系
        points_world_homo = np.hstack([points_world, np.ones((len(points_world), 1))])
        points_cam = (T_w2c @ points_world_homo.T).T[:, :3]
        
        # 投影到图像平面
        K = self.gen.get_camera_intrinsic(frame_idx, camera_name)
        points_img_homo = (K @ points_cam.T).T
        
        # 归一化
        pixel_coords = points_img_homo[:, :2] / points_img_homo[:, 2:3]
        
        # 检查点是否在视图内
        depth_img = self.gen.get_depth_image(frame_idx, camera_name)
        H, W = depth_img.shape
        
        in_view = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < W) & \
                 (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < H) & \
                 (points_cam[:, 2] > 0)  # 在相机前方
        
        return pixel_coords.astype(np.int32), in_view
    
    def render_tracking_visualization(self, output_dir: str = "point_tracking_viz",
                                     num_keyframes: int = 5,
                                     num_sample_points: int = 500,
                                     camera_name: str = "head_camera",
                                     use_gif: bool = True):
        """
        渲染点追踪可视化
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"📊 点追踪可视化")
        print(f"   总帧数: {self.gen.num_frames}")
        print(f"   采样点数: {num_sample_points}")
        
        # 计算关键帧索引
        keyframe_indices = np.linspace(0, self.gen.num_frames - 1, num_keyframes, dtype=int)
        print(f"   关键帧: {keyframe_indices}")
        
        # 从第0帧采样点
        print(f"\n📍 从第0帧采样点...")
        points_world, pixel_0 = self.sample_foreground_points(0, camera_name, num_sample_points)
        
        if len(points_world) == 0:
            print("✗ 采样失败")
            return
        
        print(f"   ✓ 成功采样 {len(points_world)} 个点")
        
        # 用不同颜色表示点（从蓝色到红色）
        num_points = len(points_world)
        colors = []
        for i in range(num_points):
            ratio = i / max(1, num_points - 1)
            # HSV色彩空间：从蓝色(240°)到红色(0°)
            h = int(240 * (1 - ratio))  # Hue
            colors.append(cv2.cvtColor(np.uint8([[[h, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        
        frames = []
        
        # 对每个关键帧进行处理
        for frame_idx in keyframe_indices:
            print(f"   处理帧 {frame_idx}/{self.gen.num_frames - 1}...")
            
            # 获取RGB背景
            rgb = self.get_rgb(frame_idx, camera_name)
            img = rgb.copy()
            
            # 投影点到当前帧
            pixels, in_view = self.project_points_to_frame(points_world, frame_idx, camera_name)
            
            # 绘制轨迹
            # 1. 绘制当前帧的点
            for i, (px, in_v) in enumerate(zip(pixels, in_view)):
                if in_v:
                    cv2.circle(img, tuple(px), 3, (int(colors[i][0]), int(colors[i][1]), int(colors[i][2])), -1)
            
            # 2. 如果不是第0帧，从前一个关键帧画轨迹线
            if frame_idx > keyframe_indices[0]:
                frame_position = np.where(keyframe_indices == frame_idx)[0][0]
                prev_frame_idx = keyframe_indices[frame_position - 1]
                prev_pixels, prev_in_view = self.project_points_to_frame(points_world, prev_frame_idx, camera_name)
                
                for i, (cur_px, prev_px, cur_v, prev_v) in enumerate(zip(pixels, prev_pixels, in_view, prev_in_view)):
                    if cur_v and prev_v:
                        cv2.line(img, tuple(prev_px), tuple(cur_px), 
                                (int(colors[i][0]), int(colors[i][1]), int(colors[i][2])), 1)
            
            # 添加文字标注
            cv2.putText(img, f"Frame {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (0, 255, 0), 2)
            cv2.putText(img, f"Points: {np.sum(in_view)}/{num_points}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            frames.append(img)
            
            # 保存单帧
            frame_path = output_dir / f"frame_{frame_idx:03d}.png"
            cv2.imwrite(str(frame_path), img)
        
        print(f"\n✓ 已生成 {len(frames)} 帧")
        
        # 生成GIF
        if use_gif:
            gif_path = output_dir / "point_tracking.gif"
            print(f"\n🎬 生成GIF: {gif_path}")
            
            pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=300,
                loop=0
            )
            print(f"   ✓ GIF大小: {gif_path.stat().st_size / 1024:.1f} KB")
        
        # 生成MP4视频
        try:
            video_path = output_dir / "point_tracking.mp4"
            print(f"\n🎬 生成MP4: {video_path}")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 5.0, 
                                (frames[0].shape[1], frames[0].shape[0]))
            
            for f in frames:
                out.write(f)
            out.release()
            print(f"   ✓ MP4大小: {video_path.stat().st_size / 1024:.1f} KB")
        except Exception as e:
            print(f"   ⚠ MP4生成失败: {e}")
        
        print(f"\n✅ 输出文件在: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="点追踪可视化")
    parser.add_argument("hdf5_path", help="HDF5文件路径")
    parser.add_argument("--output", "-o", default="point_tracking_viz",
                       help="输出目录 (default: point_tracking_viz)")
    parser.add_argument("--num-keyframes", type=int, default=5,
                       help="关键帧数 (default: 5)")
    parser.add_argument("--num-points", type=int, default=500,
                       help="采样点数 (default: 500)")
    parser.add_argument("--camera", default="head_camera",
                       help="相机名称 (default: head_camera)")
    parser.add_argument("--no-gif", action="store_true",
                       help="不生成GIF")
    
    args = parser.parse_args()
    
    viz = PointTrackingVisualizer(args.hdf5_path)
    viz.render_tracking_visualization(
        output_dir=args.output,
        num_keyframes=args.num_keyframes,
        num_sample_points=args.num_points,
        camera_name=args.camera,
        use_gif=not args.no_gif
    )


if __name__ == "__main__":
    main()
