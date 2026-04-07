"""
快速测试脚本 - 验证SceneFlow生成功能

使用方法：
    python test_sceneflow.py <hdf5_path>

示例：
    python test_sceneflow.py data/pick_diverse_bottles/aloha-agilex_clean_50/data/episode0.hdf5
"""

import sys
import os
import h5py
import numpy as np
from pathlib import Path


def inspect_hdf5(hdf5_path: str):
    """检查HDF5文件的结构"""
    print("\n" + "="*60)
    print("HDF5文件结构检查")
    print("="*60)
    
    with h5py.File(hdf5_path, 'r') as f:
        def print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  ✓ {name}: shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  📁 {name}/")
        
        print(f"\n文件: {hdf5_path}")
        f.visititems(print_structure)
        
        # 检查关键数据
        print(f"\n关键数据检查:")
        
        # 检查深度
        depth_found = False
        for key in ['observation/head_camera/depth', 'observation/front_camera/depth', 'observation/left_camera/depth', 
                    'head_camera/depth', 'depth']:
            if key in f:
                print(f"  ✓ 深度数据: {key} - shape={f[key].shape}")
                depth_found = True
                break
        if not depth_found:
            print(f"  ⚠ 未找到深度数据")
        
        # 检查相机参数
        param_found = False
        for key in ['observation/head_camera/intrinsic_cv', 'observation/front_camera/intrinsic_cv',
                    'head_camera/intrinsic_cv', 'camera_intrinsic']:
            if key in f:
                print(f"  ✓ 相机内参: {key}")
                param_found = True
                break
        if not param_found:
            print(f"  ⚠ 未找到相机内参")
        
        # 检查相机外参
        extr_found = False
        for key in ['observation/head_camera/cam2world_gl', 'observation/front_camera/cam2world_gl',
                    'head_camera/cam2world_gl', 'camera_extrinsic']:
            if key in f:
                print(f"  ✓ 相机外参: {key}")
                extr_found = True
                break
        if not extr_found:
            print(f"  ⚠ 未找到相机外参")
        
        # 检查分割
        seg_found = False
        for key in ['observation/head_camera/mesh_segmentation', 'observation/head_camera/actor_segmentation',
                    'head_camera/mesh_segmentation', 'segmentation']:
            if key in f:
                print(f"  ✓ 分割数据: {key}")
                seg_found = True
                break
        if not seg_found:
            print(f"  ⚠ 未找到分割数据")


def try_generate_sceneflow(hdf5_path: str):
    """尝试生成SceneFlow"""
    print("\n" + "="*60)
    print("测试SceneFlow生成")
    print("="*60)
    
    try:
        from generate_sceneflow import SceneFlowGenerator
        
        output_dir = os.path.join(os.path.dirname(hdf5_path), "sceneflow_test")
        
        print(f"\n正在生成SceneFlow...")
        gen = SceneFlowGenerator(hdf5_path)
        gen.load_hdf5()
        
        # 获取相机参数
        print(f"\n获取相机参数...")
        K = gen.get_camera_intrinsic(0, "head_camera")
        T_c2w = gen.get_camera_extrinsic(0, "head_camera")
        depth = gen.get_depth_image(0, "head_camera")
        seg = gen.get_segmentation(0, "head_camera")
        
        print(f"✓ 相机内参 K:\n{K}")
        print(f"✓ 相机外参 T_c2w:\n{T_c2w}")
        print(f"✓ 深度图: {depth.shape}")
        print(f"✓ 分割图: {seg.shape}")
        
        # 测试逆投影
        print(f"\n测试逆投影...")
        pc = gen.depth_to_pointcloud(depth, K, T_c2w)
        print(f"✓ 生成的点云: {pc.shape}")
        
        # 完整流程
        print(f"\n执行完整流程...")
        gen.generate_sceneflow(output_dir, camera_name="head_camera")
        
        # 检查输出
        print(f"\n✅ 生成成功！输出目录: {output_dir}")
        for f in Path(output_dir).glob("*.npy"):
            arr = np.load(f)
            print(f"  ✓ {f.name}: {arr.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    if len(sys.argv) < 2:
        print("用法: python test_sceneflow.py <hdf5_path>")
        print("\n示例:")
        print("  python test_sceneflow.py data/pick_diverse_bottles/aloha-agilex_clean_50/data/episode0.hdf5")
        sys.exit(1)
    
    hdf5_path = sys.argv[1]
    
    if not os.path.exists(hdf5_path):
        print(f"❌ 文件不存在: {hdf5_path}")
        sys.exit(1)
    
    # 1. 检查HDF5结构
    inspect_hdf5(hdf5_path)
    
    # 2. 尝试生成SceneFlow
    success = try_generate_sceneflow(hdf5_path)
    
    if success:
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        print("\n后续步骤:")
        print("1. 生成完整的SceneFlow:")
        print(f"   python generate_sceneflow.py {hdf5_path}")
        print(f"\n2. 可视化结果:")
        output_dir = os.path.join(os.path.dirname(hdf5_path), "sceneflow_test")
        print(f"   python visualize_sceneflow.py {output_dir}")
    else:
        print("\n❌ 测试失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
