"""Pre-processor

The pre-processing modules are described in this file.
"""

import cv2


def image_preprocess(x, norm_coef=[0.0, 1.0]):
   """Image Pre-process
   
   Args:
       x (numpy.ndarray): input images [N, H, W, C], Channel=[3:RGB or 1:Grayscale]
       norm_coef (list of float): coefficient for normalization [alpha, beta]
           - (normalized value) = (x - alpha) / beta
   """
   
   # --- normalization ---
   y = (x - norm_coef[0]) / norm_coef[1]
   
   return y
   
   