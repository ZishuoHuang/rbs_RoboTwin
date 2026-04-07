"""
SceneFlow点云动画渲染脚本
将点云序列保存为MP4视频，显示物体运动

使用方法：
    python render_sceneflow_video.py <sceneflow_dir> [--output video.mp4] [--fps 10]
    
示例：
    python render_sceneflow_video.py data/pick_diverse_bottles/aloha-agilex_clean_50/sceneflow_episode0
"""

import numpy as np
from pathlib import Path
import os
import sys
from typing import List

try:
    import cv2
except ImportError:
    print("需要安装 opencv: pip install opencv-python")
    sys.exit(1)


class SceneFlowVideoRenderer:
    """SceneFlow视频渲染器"""
    
    def __init__(self, sceneflow_dir: str):
        """
        Args:
            sceneflow_dir: SceneFlow目录
        """
        self.sceneflow_dir = Path(sceneflow_dir)
        self.pointcloud_frame0 = None
        self.delta_trajectories = []
        self.segmentation = None
        
    def load_data(self):
        """加载SceneFlow数据"""
        print("加载数据...")
        
        # 加载点云
        pc_path = self.sceneflow_dir / "pointcloud_frame0.npy"
        self.pointcloud_frame0 = np.load(pc_path)
        print(f"✓ 点云: {self.pointcloud_frame0.shape}")
        
        # 加载分割
        seg_path = self.sceneflow_dir / "segmentation_frame0.npy"
        self.segmentation = np.load(seg_path)
        print(f"✓ 分割: {self.segmentation.shape}")
        
        # 加载所有delta轨迹
        sceneflow_files = sorted(self.sceneflow_dir.glob("sceneflow_*.npy"))
        for f in sceneflow_files:
            delta = np.load(f)
            self.delta_trajectories.append(delta)
        
        print(f"✓ 加载 {len(self.delta_trajectories)} 条轨迹")
    
    def create_color_map(self):
        """创建物体颜色映射"""
        unique_ids = np.unique(self.segmentation)
        
        colors = [
            (255, 0, 0),      # 红
            (0, 255, 0),      # 绿
            (0, 0, 255),      # 蓝
            (255, 255, 0),    # 黄
            (255, 0, 255),    # 紫
            (0, 255, 255),    # 青
            (255, 165, 0),    # 橙
            (128, 0, 128),    # 深紫
        ]
        
        color_map = {}
        for obj_id in unique_ids:
            color_idx = int(obj_id) % len(colors)
            color_map[int(obj_id)] = colors[color_idx]
        
        return color_map
    
    def project_to_image(self, points_3d: np.ndarray, width: int = 1024, 
                        height: int = 768, focal_length: float = 800) -> np.ndarray:
        """
        将3D点投影到2D图像平面
        
        Args:
            points_3d: (N, 3) 三维点
            width: 图像宽度
            height: 图像高度
            focal_length: 焦距
            
        Returns:
            image: (height, width, 3) 图像
            pixel_size: 每个点的像素大小
        """
        # 创建空白图像
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 标准化点到[-1, 1]范围
        points_min = points_3d.min(axis=0)
        points_max = points_3d.max(axis=0)
        points_range = points_max - points_min
        
        # 确保不为0
        points_range[points_range == 0] = 1
        
        points_norm = (points_3d - points_min) / points_range * 2 - 1
        
        # 简单的透视投影
        # 假设相机在 z=0
        z = points_3d[:, 2] - points_min[2]
        z = z / (points_range[2] + 1e-6) * 2  # normalize depth for size
        
        # 投影到图像
        x_2d = (points_norm[:, 0] + 1) * width / 2
        y_2d = (points_norm[:, 1] + 1) * height / 2
        
        # 确保在图像范围内
        valid_mask = (x_2d >= 0) & (x_2d < width) & (y_2d >= 0) & (y_2d < height)
        
        x_2d = x_2d[valid_mask].astype(np.int32)
        y_2d = y_2d[valid_mask].astype(np.int32)
        
        # 根据深度调整点的大小
        pixel_size = np.clip(z[valid_mask] * 5 + 1, 1, 10).astype(np.int32)
        
        return image, x_2d, y_2d, pixel_size, valid_mask
    
    def render_frame(self, points_3d: np.ndarray, colors: np.ndarray,
                    width: int = 1024, height: int = 768,
                    title: str = "SceneFlow") -> np.ndarray:
        """
        渲染单帧
        
        Args:
            points_3d: (N, 3) 三维点
            colors: (N, 3) RGB颜色
            width: 图像宽度
            height: 图像高度
            title: 标题
            
        Returns:
            image: (height, width, 3) 渲染的图像
        """
        image, x_2d, y_2d, pixel_size, valid_mask = self.project_to_image(
            points_3d, width, height
        )
        
        # 绘制有效的点
        valid_colors = colors[valid_mask]
        
        for i in range(len(x_2d)):
            x, y, size = int(x_2d[i]), int(y_2d[i]), int(pixel_size[i])
            color = tuple(int(c) for c in valid_colors[i])
            
            # 绘制点
            cv2.circle(image, (x, y), size, color, -1)
        
        # 添加标题
        cv2.putText(image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return image
    
    def render_video(self, output_path: str = "sceneflow.mp4", fps: int = 10,
                    width: int = 1024, height: int = 768, 
                    interpolate: bool = True, num_interp: int = 5,
                    video_format: str = "avi"):
        """
        渲染视频
        
        Args:
            output_path: 输出视频路径
            fps: 帧率
            width: 图像宽度
            height: 图像高度
            interpolate: 是否在关键帧之间插值
            num_interp: 每两个关键帧之间插值的帧数
            video_format: 视频格式 ('avi' 或 'mp4')
        """
        print(f"\n渲染视频: {output_path}")
        
        # 创建视频writer - 使用MJPEG编码（最兼容）
        if video_format.lower() == "avi":
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            if not output_path.endswith('.avi'):
                output_path = output_path.replace('.mp4', '.avi')
        else:  # mp4
            # 尝试多种编码
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if not output_path.endswith('.mp4'):
                output_path = output_path + '.mp4'
        
        if interpolate:
            actual_fps = min(fps, len(self.delta_trajectories) * (num_interp + 1) // 5)
        else:
            actual_fps = fps
        
        out = cv2.VideoWriter(output_path, fourcc, actual_fps, (width, height))
        
        if not out.isOpened():
            print("⚠ 警告: VideoWriter不兼容，尝试MJPEG编码...")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            output_path = output_path.replace('.mp4', '.avi')
            out = cv2.VideoWriter(output_path, fourcc, actual_fps, (width, height))
            
            if not out.isOpened():
                print("❌ 无法创建视频writer，尝试保存为图像序列...")
                return False
        
        # 创建颜色映射
        color_map = self.create_color_map()
        
        # 渲染每一帧
        frames_to_render = []
        frame_titles = []
        
        if interpolate and len(self.delta_trajectories) > 1:
            # 在关键帧之间插值
            print("在关键帧之间插值...")
            
            for frame_idx, delta in enumerate(self.delta_trajectories):
                # 当前关键帧
                points_current = self.pointcloud_frame0 + delta
                
                # 添加原始关键帧
                frames_to_render.append(points_current)
                frame_titles.append(f"Keyframe {frame_idx}")
                
                # 如果不是最后一帧，添加插值帧
                if frame_idx < len(self.delta_trajectories) - 1:
                    delta_next = self.delta_trajectories[frame_idx + 1]
                    points_next = self.pointcloud_frame0 + delta_next
                    
                    # 线性插值
                    for interp_idx in range(1, num_interp + 1):
                        alpha = interp_idx / (num_interp + 1)
                        points_interp = points_current * (1 - alpha) + points_next * alpha
                        frames_to_render.append(points_interp)
                        frame_titles.append(f"Interp {frame_idx}->{frame_idx+1} ({interp_idx}/{num_interp})")
        else:
            # 只渲染关键帧
            for frame_idx, delta in enumerate(self.delta_trajectories):
                points = self.pointcloud_frame0 + delta
                frames_to_render.append(points)
                frame_titles.append(f"Frame {frame_idx}")
        
        print(f"总帧数: {len(frames_to_render)}")
        
        # 写入每一帧
        for frame_idx, (points, title) in enumerate(zip(frames_to_render, frame_titles)):
            print(f"  渲染帧 {frame_idx + 1}/{len(frames_to_render)}...", end='\r')
            
            # 为每个点着色（基于分割）
            # 需要找出每个3D点对应的分割ID
            # 简化：按照点云顺序着色，但分割是(H,W)格式
            
            # 创建颜色数组
            colors = np.zeros((len(points), 3), dtype=np.uint8)
            
            # 尝试将分割映射到点
            # 注：这里需要知道点如何来自分割图
            # 暂时用随机颜色或单一颜色
            
            # 方案：使用点的Z坐标着色（深度着色）
            z_min = points[:, 2].min()
            z_max = points[:, 2].max()
            if z_max > z_min:
                z_norm = (points[:, 2] - z_min) / (z_max - z_min)
            else:
                z_norm = np.ones(len(points))
            
            # HSV转RGB：H=z_norm*180, S=255, V=255
            for i in range(len(points)):
                h = int(z_norm[i] * 180)
                hsv = np.uint8([[[h, 255, 255]]])
                rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
                colors[i] = rgb
            
            # 渲染帧
            image = self.render_frame(points, colors, width, height, title)
            
            # 写入视频
            out.write(image)
        
        print()  # 换行
        out.release()
        print(f"✓ 视频保存: {output_path}")
        return True
    
    def render_sceneflow_video(self, output_path: str = None, fps: int = 10,
                              video_format: str = "avi"):
        """
        完整的视频生成流程
        
        Args:
            output_path: 输出路径
            fps: 帧率
            video_format: 视频格式 ('avi' 或 'mp4')
        """
        if output_path is None:
            ext = ".avi" if video_format.lower() == "avi" else ".mp4"
            output_path = self.sceneflow_dir / f"sceneflow_animation{ext}"
        
        print("\n" + "="*60)
        print("SceneFlow视频渲染")
        print("="*60)
        
        # 加载数据
        self.load_data()
        
        # 创建视频
        self.render_video(str(output_path), fps=fps, interpolate=True, num_interp=3,
                         video_format=video_format)
        
        if os.path.exists(output_path):
            print(f"\n✅ 完成！视频已保存到: {output_path}")
            print(f"   文件大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
        else:
            print(f"❌ 视频生成失败")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="渲染SceneFlow视频")
    parser.add_argument("sceneflow_dir", type=str, help="SceneFlow目录")
    parser.add_argument("--output", type=str, default=None, help="输出视频路径")
    parser.add_argument("--fps", type=int, default=10, help="帧率")
    parser.add_argument("--width", type=int, default=1024, help="视频宽度")
    parser.add_argument("--height", type=int, default=768, help="视频高度")
    parser.add_argument("--format", type=str, default="avi", 
                       choices=["avi", "mp4"], help="输出格式 (avi更兼容)")
    parser.add_argument("--no-interpolate", action="store_true", help="禁用关键帧插值")
    
    args = parser.parse_args()
    
    renderer = SceneFlowVideoRenderer(args.sceneflow_dir)
    
    if args.output is None:
        ext = ".avi" if args.format.lower() == "avi" else ".mp4"
        args.output = os.path.join(args.sceneflow_dir, f"sceneflow_animation{ext}")
    
    renderer.render_sceneflow_video(args.output, fps=args.fps, video_format=args.format)


if __name__ == "__main__":
    main()
