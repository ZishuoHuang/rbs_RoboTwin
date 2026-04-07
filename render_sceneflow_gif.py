"""
SceneFlow GIF生成脚本
将点云序列保存为动画GIF，可用任何浏览器打开

使用方法：
    python render_sceneflow_gif.py <sceneflow_dir> [--output animation.gif]
    
示例：
    python render_sceneflow_gif.py data/pick_diverse_bottles/aloha-agilex_clean_50/sceneflow_episode0
"""

import numpy as np
from pathlib import Path
import os
import sys
from typing import List

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("需要安装 Pillow: pip install Pillow")
    sys.exit(1)


class SceneFlowGIFRenderer:
    """SceneFlow GIF渲染器"""
    
    def __init__(self, sceneflow_dir: str):
        """
        Args:
            sceneflow_dir: SceneFlow目录
        """
        self.sceneflow_dir = Path(sceneflow_dir)
        self.pointcloud_frame0 = None
        self.delta_trajectories = []
        self.segmentation = None
        self.point_seg_ids = None
        
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

        # 加载逐点分割（若存在）
        point_seg_path = self.sceneflow_dir / "point_seg_ids_frame0.npy"
        if point_seg_path.exists():
            self.point_seg_ids = np.load(point_seg_path).astype(np.int32)
            print(f"✓ 逐点分割: {self.point_seg_ids.shape}")
        else:
            self.point_seg_ids = None
            print("⚠ 未找到 point_seg_ids_frame0.npy，将退化为深度着色")
        
        # 加载所有delta轨迹
        sceneflow_files = sorted(self.sceneflow_dir.glob("sceneflow_*.npy"))
        for f in sceneflow_files:
            delta = np.load(f)
            self.delta_trajectories.append(delta)
        
        print(f"✓ 加载 {len(self.delta_trajectories)} 条轨迹")
    
    def project_to_image(self, points_3d: np.ndarray, width: int = 800,
                        height: int = 600) -> tuple:
        """
        将3D点投影到2D图像平面
        
        Args:
            points_3d: (N, 3) 三维点
            width: 图像宽度
            height: 图像高度
            
        Returns:
            (image_array, x_2d, y_2d, depth, valid_mask)
        """
        # 创建空白图像（白色背景）
        image_array = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 标准化点到[-1, 1]范围
        points_min = points_3d.min(axis=0)
        points_max = points_3d.max(axis=0)
        points_range = points_max - points_min
        
        # 确保不为0
        points_range[points_range == 0] = 1
        
        points_norm = (points_3d - points_min) / points_range * 2 - 1
        
        # 深度信息用于大小
        z = points_3d[:, 2] - points_min[2]
        z = z / (points_range[2] + 1e-6)
        
        # 投影到图像
        x_2d = (points_norm[:, 0] + 1) * width / 2
        y_2d = (points_norm[:, 1] + 1) * height / 2
        
        # 确保在图像范围内
        valid_mask = (x_2d >= 0) & (x_2d < width) & (y_2d >= 0) & (y_2d < height)
        
        x_2d = x_2d[valid_mask].astype(np.int32)
        y_2d = y_2d[valid_mask].astype(np.int32)
        z = z[valid_mask]
        
        return image_array, x_2d, y_2d, z, valid_mask
    
    def _build_seg_color_lut(self):
        """为分割id建立稳定颜色映射。"""
        if self.point_seg_ids is None:
            return {}
        unique_ids = np.unique(self.point_seg_ids)
        lut = {}
        for sid in unique_ids:
            s = int(sid)
            # 稳定伪随机配色（同一物体同色）
            r = (s * 37 + 53) % 256
            g = (s * 97 + 29) % 256
            b = (s * 17 + 131) % 256
            lut[s] = (int(r), int(g), int(b), 220)
        return lut

    def render_frame_pil(self, points_3d: np.ndarray, width: int = 800,
                        height: int = 600, title: str = "SceneFlow", color_mode: str = "depth",
                        seg_color_lut: dict = None, seg_point_size: int = 1) -> Image.Image:
        """
        用PIL渲染单帧
        
        Args:
            points_3d: (N, 3) 三维点
            width: 图像宽度
            height: 图像高度
            title: 标题
            
        Returns:
            PIL Image对象
        """
        # 创建图像
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image, 'RGBA')
        
        # 投影
        _, x_2d, y_2d, z, valid_mask = self.project_to_image(
            points_3d, width, height
        )
        
        # 按深度排序（远的先画，近的后画）
        depth_order = np.argsort(-z)
        
        x_2d_sorted = x_2d[depth_order]
        y_2d_sorted = y_2d[depth_order]
        z_sorted = z[depth_order]
        
        if color_mode == "segmentation" and self.point_seg_ids is not None and len(self.point_seg_ids) == len(points_3d):
            seg_valid = self.point_seg_ids[valid_mask]
            seg_valid_sorted = seg_valid[depth_order]
            for x, y, sid in zip(x_2d_sorted, y_2d_sorted, seg_valid_sorted):
                color = seg_color_lut.get(int(sid), (200, 200, 200, 220))
                size = max(1, int(seg_point_size))
                draw.ellipse([x - size, y - size, x + size, y + size], fill=color, outline=None)
        else:
            # 绘制点（深度着色）
            for x, y, depth in zip(x_2d_sorted, y_2d_sorted, z_sorted):
                h = int(depth * 180)
                r = int(255 * (1 - abs(depth - 0.5) * 2))
                g = int(255 * depth)
                b = int(255 * (1 - depth))

                color = (r, g, b, 200)
                size = max(1, int(depth * 3) + 1)
                draw.ellipse([x - size, y - size, x + size, y + size], fill=color, outline=None)
        
        # 添加标题
        try:
            draw.text((10, 10), title, fill='black')
        except:
            # 如果字体加载失败，不添加标题
            pass
        
        return image
    
    def render_gif(self, output_path: str = "sceneflow.gif", 
                   width: int = 800, height: int = 600,
                   duration: int = 200, interpolate: bool = True,
                   num_interp: int = 3, color_mode: str = "depth",
                   seg_point_size: int = 1):
        """
        渲染GIF
        
        Args:
            output_path: 输出GIF路径
            width: 图像宽度
            height: 图像高度
            duration: 每帧持续时间（毫秒）
            interpolate: 是否在关键帧之间插值
            num_interp: 每两个关键帧之间插值的帧数
        """
        print(f"\n渲染GIF: {output_path}")
        
        frames = []
        frame_titles = []
        seg_color_lut = self._build_seg_color_lut()
        
        if interpolate and len(self.delta_trajectories) > 1:
            # 在关键帧之间插值
            print("在关键帧之间插值...")
            
            for frame_idx, delta in enumerate(self.delta_trajectories):
                # 当前关键帧
                points_current = self.pointcloud_frame0 + delta
                
                # 添加原始关键帧
                frames.append(points_current)
                frame_titles.append(f"Keyframe {frame_idx}")
                
                # 如果不是最后一帧，添加插值帧
                if frame_idx < len(self.delta_trajectories) - 1:
                    delta_next = self.delta_trajectories[frame_idx + 1]
                    points_next = self.pointcloud_frame0 + delta_next
                    
                    # 线性插值
                    for interp_idx in range(1, num_interp + 1):
                        alpha = interp_idx / (num_interp + 1)
                        points_interp = points_current * (1 - alpha) + points_next * alpha
                        frames.append(points_interp)
                        frame_titles.append(f"Inter {frame_idx}-{frame_idx+1}")
        else:
            # 只渲染关键帧
            for frame_idx, delta in enumerate(self.delta_trajectories):
                points = self.pointcloud_frame0 + delta
                frames.append(points)
                frame_titles.append(f"Frame {frame_idx}")
        
        print(f"总帧数: {len(frames)}")
        
        # 渲染每一帧
        pil_frames = []
        for frame_idx, (points, title) in enumerate(zip(frames, frame_titles)):
            print(f"  渲染帧 {frame_idx + 1}/{len(frames)}...", end='\r')

            if color_mode == "segmentation":
                pil_image = self.render_frame_pil(
                    points,
                    width,
                    height,
                    title,
                    color_mode="segmentation",
                    seg_color_lut=seg_color_lut,
                    seg_point_size=seg_point_size,
                )
            else:
                pil_image = self.render_frame_pil(points, width, height, title, color_mode="depth")
            pil_frames.append(pil_image)
        
        print()  # 换行
        
        # 保存为GIF
        print(f"保存GIF...")
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0  # 无限循环
        )
        
        print(f"✓ GIF保存: {output_path}")
        return True
    
    def render_sceneflow_gif(self, output_path: str = None, 
                            width: int = 800, height: int = 600,
                            duration: int = 200, interpolate: bool = True,
                            num_interp: int = 2, color_mode: str = "depth",
                            seg_point_size: int = 1):
        """
        完整的GIF生成流程
        
        Args:
            output_path: 输出路径
            width: 图像宽度
            height: 图像高度
            duration: 每帧时长（毫秒）
        """
        if output_path is None:
            output_path = self.sceneflow_dir / "sceneflow_animation.gif"
        
        print("\n" + "="*60)
        print("SceneFlow GIF渲染")
        print("="*60)
        
        # 加载数据
        self.load_data()
        
        # 创建GIF
        self.render_gif(
            str(output_path),
            width=width,
            height=height,
            duration=duration,
            interpolate=interpolate,
            num_interp=num_interp,
            color_mode=color_mode,
            seg_point_size=seg_point_size,
        )
        
        if os.path.exists(output_path):
            print(f"\n✅ 完成！GIF已保存到: {output_path}")
            print(f"   文件大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
            print(f"\n💡 提示: 用任何浏览器打开即可查看动画！")
        else:
            print(f"❌ GIF生成失败")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="渲染SceneFlow GIF")
    parser.add_argument("sceneflow_dir", type=str, help="SceneFlow目录")
    parser.add_argument("--output", type=str, default=None, help="输出GIF路径")
    parser.add_argument("--width", type=int, default=800, help="GIF宽度")
    parser.add_argument("--height", type=int, default=600, help="GIF高度")
    parser.add_argument("--duration", type=int, default=200, 
                       help="每帧持续时间（毫秒）")
    parser.add_argument("--no-interpolate", action="store_true",
                        help="禁用关键帧插值（用于全过程帧时建议开启）")
    parser.add_argument("--num-interp", type=int, default=2,
                        help="关键帧间插值数量")
    parser.add_argument("--color-mode", type=str, default="depth",
                        choices=["depth", "segmentation"],
                        help="点云着色模式：depth 或 segmentation")
    parser.add_argument("--seg-point-size", type=int, default=1,
                        help="segmentation着色时的点半径像素，增大可让分块更明显")
    
    args = parser.parse_args()
    
    renderer = SceneFlowGIFRenderer(args.sceneflow_dir)
    
    if args.output is None:
        args.output = os.path.join(args.sceneflow_dir, "sceneflow_animation.gif")
    
    renderer.render_sceneflow_gif(
        args.output,
        width=args.width,
        height=args.height,
        duration=args.duration,
        interpolate=not args.no_interpolate,
        num_interp=args.num_interp,
        color_mode=args.color_mode,
        seg_point_size=args.seg_point_size,
    )


if __name__ == "__main__":
    main()
