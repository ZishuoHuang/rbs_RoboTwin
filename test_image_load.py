import h5py
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt

data_path = '/home/CNS2026497693/RoboTwin/data/pick_diverse_bottles/aloha-agilex_clean_50/data_done/episode0.hdf5'
with h5py.File(data_path, 'r') as f:
    frame_idx = 50
    rgb_bytes = f['observation/head_camera/rgb'][frame_idx]
    
    # Use PIL to read the image bytes properly
    rgb_img = Image.open(io.BytesIO(rgb_bytes))
    rgb_img = np.array(rgb_img)
    print(f"Image shape: {rgb_img.shape}, Data style: RGB")
    
    plt.imshow(rgb_img)
    plt.savefig('test_rgb_only.png')
