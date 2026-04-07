import h5py
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

data_path = '/home/CNS2026497693/RoboTwin/data/pick_diverse_bottles/aloha-agilex_clean_50/data_done/episode0.hdf5'

with h5py.File(data_path, 'r') as f:
    # Let's extract frame 50 as an example
    frame_idx = 50
    
    # 1. RGB Image (compressed)
    rgb_bytes = f['observation/head_camera/rgb'][frame_idx]
    rgb_np = np.frombuffer(rgb_bytes, dtype=np.uint8)
    # The image was saved directly as RGB via cv2, so decoding it with cv2 yields actual RGB array directly (it thinks it's BGR, but channels are RGB)
    rgb_img = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR) 
    # cv2.cvtColor(..., cv2.COLOR_BGR2RGB) is removed because they were already in RGB order.
    
    # 2. Depth Image
    depth_img = f['observation/head_camera/depth'][frame_idx]
    
    # 3. Actor Segmentation (Instant segmentation etc.)
    actor_seg = f['observation/head_camera/actor_segmentation'][frame_idx]
    
    # 4. Mesh Segmentation
    mesh_seg = f['observation/head_camera/mesh_segmentation'][frame_idx]
    
    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb_img)
    axes[0].set_title('RGB')
    
    axes[1].imshow(depth_img, cmap='plasma')
    axes[1].set_title('Depth')
    
    axes[2].imshow(actor_seg)
    axes[2].set_title('Actor Segmentation')
    
    axes[3].imshow(mesh_seg)
    axes[3].set_title('Mesh Segmentation')
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig('visualize_frame.png')
    print("Saved visualization to visualize_frame.png")

