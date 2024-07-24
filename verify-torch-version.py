import torch
import numpy as np
import cv2
from easydict import EasyDict

print("PyTorch version:", torch.__version__)
print("NumPy version:", np.__version__)
print("OpenCV version:", cv2.__version__)
print("EasyDict version:", EasyDict)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Device name:", torch.cuda.get_device_name(0))
