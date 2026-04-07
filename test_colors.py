import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt

data_path = '/home/CNS2026497693/RoboTwin/data/pick_diverse_bottles/aloha-agilex_clean_50/data_done/episode0.hdf5'

with h5py.File(data_path, 'r') as f:
    frame_idx = 50
    rgb_bytes = f['observation/head_camera/rgb'][frame_idx]
    
    rgb_np = np.frombuffer(rgb_bytes, dtype=np.uint8)
    image_decoded = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)
    
    # 1. As decoded (if it was already RGB when saved, cv2 thinks it's BGR)
    img_as_decoded = image_decoded
    
    # 2. Swap channels (BGR to RGB or vice versa)
    img_swapped = cv2.cvtColor(image_decoded, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img_as_decoded)
    axes[0].set_title('As Decoded (cv2 BGR)')
    
    axes[1].imshow(img_swapped)
    axes[1].set_title('Swapped (cv2 RGB)')
    
    plt.tight_layout()
    plt.savefig('test_colors.png')
