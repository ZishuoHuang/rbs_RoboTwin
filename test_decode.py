import h5py
import cv2
import numpy as np

file_path = "data/pick_diverse_bottles/aloha-agilex_clean_50/data/episode1.hdf5"
with h5py.File(file_path, "r") as f:
    rgb_bytes = f["observation/left_camera/rgb"][10]
    
# Decode
img_array = np.frombuffer(rgb_bytes, dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# Fix OpenCV's default BGR format to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

cv2.imwrite("test_decoded.jpg", img_rgb)
print("Saved test_decoded.jpg successfully with correct channels!")