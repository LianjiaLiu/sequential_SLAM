import os
import numpy as np

SIFT  = 0
ORB   = 1
BRIEF = 2 
combined = 3
target_obj = 'box'
# extract_method = 
image_dir = f'D:/Gitte_Belly/Desktop/test/sfm_test/data/{target_obj}/images/'
# corres_dir = f'D:/Gitte_Belly/Desktop/test/sfm_test/data/{target_obj}/corres/{extract_method}_corres/'

MRT = 0.7


K = np.array([[927.494, 0.0, 960.0], [0.0, 927.494, 540.0], [0.0, 0.0, 1.0]])

#used for outlier removal
x = 0.5
y = 1