import os
import cv2
import sys
sys.path.append("..")
import math
import config
import collections
import numpy as np
import matplotlib.pyplot as plt
from src.match import match
# from mayavi import mlab
from scipy.linalg import lstsq
#from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares
import torch

##########################
#Detect and match features
##########################
def extract_features(image_names, method_name = 'sift'):
    """
    return the key points, descriptors and respective colors of the images
    """
    if method_name=='sift': extractor = cv2.SIFT_create(0, 3, 0.04, 10)
    elif method_name=='brief':  
        fast = cv2.FastFeatureDetector_create()
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    elif method_name=='orb': extractor = cv2.ORB_create()
    key_points_for_all = []
    descriptor_for_all = []
    colors_for_all = []

    for image_name in image_names:
        image = cv2.imread(os.path.join(config.image_dir, image_name))
        image_color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            continue
        if method_name=='brief':
            key_points = fast.detect(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
            key_points, descriptor = brief.compute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), key_points)
        else: key_points, descriptor = extractor.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
        
        if len(key_points) <= 10:
            continue
        
        key_points_for_all.append(key_points)
        descriptor_for_all.append(descriptor)

        colors = np.zeros((len(key_points), 3))
        for i, key_point in enumerate(key_points):
            p = key_point.pt
            colors[i] = image_color[int(p[1])][int(p[0])]         
        colors_for_all.append(colors)

    print(f"shape for one image: {image.shape}")
    return key_points_for_all, descriptor_for_all, colors_for_all

def match_features(query, train,device):
    # ----ours----#
    matches = match(
                        descriptors1= torch.tensor(query, device = device,dtype = torch.float32),
                        descriptors2= torch.tensor(train, device = device,dtype = torch.float32),
                        device=device,
                        ratio=0.72,
                        threshold=380

                    ).detach().cpu().numpy()
    return np.array(matches)

def match_all_features(descriptor_for_all, device):
    """
    for sequential images (img[i],img[i+1])
    """
    matches_for_all = []
    for i in range(len(descriptor_for_all) - 1):
        matches = match_features(descriptor_for_all[i], descriptor_for_all[i + 1], device=device)
        assert matches.shape[0] > 0, "No matches found in match_all_features()" # would break the algorithm because we are reconstructing sequentially
        matches_for_all.append(matches)

    return matches_for_all


######################
#Find the corresponding camera rotation angle and camera translation between images
######################
def find_transform(K, p1, p2):  
    """
    Return the camera pose between two images, plus mask of inliers and essential matrix
    Input:
    -K: 3*3, intrinsic matrix
    -mask: set to 0 for outliers and to 1 for the other points (inliers)
    """    
    focal_length    = 0.5 * (K[0, 0] + K[1, 1])
    principle_point = (K[0, 2], K[1, 2])
    E,mask          = cv2.findEssentialMat(p1, p2, focal_length, principle_point, cv2.RANSAC, 0.999, 1.0)
    cameraMatrix    = np.array([[focal_length, 0, principle_point[0]], [0, focal_length, principle_point[1]], [0, 0, 1]])
    pass_count, R, T, mask = cv2.recoverPose(E, p1, p2, cameraMatrix, mask)
    
    return R, T, mask, E

def get_matched_points(p1, p2, matches):
    """
    get the matched 2d points coord of matching list
    Input:
    -matches:  [num_of_matches, 2]
    """
    assert len(matches[0]) ==2, "Wrong dimension of single match in get_matched_points()"

    src_pts = []
    dst_pts = []
    for m in matches:
        src_pts.append(p1[m[0]].pt)
        dst_pts.append(p2[m[1]].pt)
    assert np.array(src_pts).shape == (len(matches),2), "Wrong dimension of matched point list in get_matched_points()"

    return np.array(src_pts), np.array(dst_pts)

def get_matched_colors(c1, c2, matches):
    
    color_src_pts = []
    color_dst_pts = []
    for m in matches:
        color_src_pts.append(c1[m[0]])
        color_dst_pts.append(c2[m[1]])
    assert np.array(color_src_pts).shape == (len(matches),3), "Wrong dimension of matched color list in get_matched_colors()"

    return np.array(color_src_pts), np.array(color_dst_pts)

def maskout_points(p1, mask):   
    """
    filter out outliers of the matched points
    """
    p1_copy = []
    for i in range(len(mask)):
        if mask[i] > 0:
            p1_copy.append(p1[i])
    
    return np.array(p1_copy)
    
def init_structure(K, key_points_for_all, colors_for_all, matches_for_all):  
    """
    Input:
    -key_points_for_all: [imgs,  num_of_keypoint, keypoint]
    -colors_for_all:     [imgs,  num_of_keypoint, 3]
    -matches_for_all:    [imgs-1, num_of_matches, 2]

    Return:
    -structure:             [num_of_3D_points, 3]
    -correspond_struct_idx: [imgs, num_of_keypoint], 
                            mapping the 2D points to 3D points, 3d_idx = correspond_struct_idx[i][2d_idx], 
                            2d_idx refers to the index of the keypoint in the i-th image
    -colors:                [num_of_3D_points, 3]
    -rotations:             [imgs, 3, 3]
    -motions:               [imgs, 3, 1]
    """
    assert len(key_points_for_all) == len(colors_for_all)==len(matches_for_all)+1, "Inconsistency of dimensions in init_structure()"

    p1, p2 = get_matched_points(key_points_for_all[0], key_points_for_all[1], matches_for_all[0])
    c1, c2 = get_matched_colors(colors_for_all[0], colors_for_all[1], matches_for_all[0])

    # camera pose
    if find_transform(K, p1, p2):
        R,T,mask, E = find_transform(K, p1, p2)
    else:
        R,T,mask, E = np.array([]), np.array([]), np.array([])
    
    p1     = maskout_points(p1, mask)
    p2     = maskout_points(p2, mask)
    colors = maskout_points(c1, mask)

    # set the first camera pose as the base
    R0        = np.eye(3, 3)
    T0        = np.zeros((3, 1))
    # 3D points
    structure = reconstruct(K, R0, T0, R, T, p1, p2) 

    rotations = [R0, R]
    motions   = [T0, T]
    correspond_struct_idx = [] 
    for key_p in key_points_for_all:
        correspond_struct_idx.append(np.ones(len(key_p)) *- 1) # -1 means no corresponding 3D point yet

    idx = 0 #index of the reconstructed 3D point list
    matches = matches_for_all[0] #shape(num_of_matches, 2) for the first two images
    for i, match in enumerate(matches):
        if mask[i] == 0: #outliers
            continue
        correspond_struct_idx[0][int(match[0])] = idx
        correspond_struct_idx[1][int(match[1])] = idx
        idx += 1

    return structure, correspond_struct_idx, colors, rotations, motions
    
#############
#3D reconstruction
#############
def reconstruct(K, R1, T1, R2, T2, p1, p2):
    """
    return the 3D coordinates of the points
    """
    proj1 = np.zeros((3, 4))
    proj2 = np.zeros((3, 4))
    proj1[0:3, 0:3] = np.float32(R1)
    proj1[:, 3]     = np.float32(T1.T)
    proj2[0:3, 0:3] = np.float32(R2)
    proj2[:, 3]     = np.float32(T2.T)
    fk    = np.float32(K)
    proj1 = np.dot(fk, proj1)
    proj2 = np.dot(fk, proj2)
    s     = cv2.triangulatePoints(proj1, proj2, p1.T, p2.T)

    structure = []
    
    for i in range(len(s[0])):
        col  = s[:, i]
        col /= col[3]
        structure.append([col[0], col[1], col[2]])
    
    return np.array(structure)

###########################
#fusing the existing structure and the new structure
###########################
def fusion_structure(matches, struct_indices, next_struct_indices, structure, next_structure, colors, next_colors):
    """
    Update new 3D points, their colors.
    Update their 3D indices in the structure_idx list of two frames.
    """
    for i,match in enumerate(matches):  
        query_idx  = match[0]
        train_idx  = match[1]
        struct_idx = struct_indices[query_idx]  
        if struct_idx >= 0:
            next_struct_indices[train_idx] = struct_idx
            continue
        structure = np.append(structure, [next_structure[i]], axis = 0)
        colors    = np.append(colors, [next_colors[i]], axis = 0)
        struct_indices[query_idx] = next_struct_indices[train_idx] = len(structure) - 1
    return struct_indices, next_struct_indices, structure, colors

def get_objpoints_and_imgpoints(matches, struct_indices, structure, key_points):
    """
    Return the reconstructed 3D points and its corresponding 2D points 
    Input:
    -key_points:     refer to the infor of images(2D points)
    -stract_indices: refer to the infor of 3D points in "world" coordinate
    """
    object_points = []
    image_points  = []
    for match in matches:
        query_idx  = match[0]
        train_idx  = match[1]
        struct_idx = struct_indices[query_idx]  
        if struct_idx < 0: #no 3D point yet
            continue
        object_points.append(structure[int(struct_idx)])
        image_points.append(key_points[train_idx].pt)
    
    return np.array(object_points), np.array(image_points)

########################
#bundle adjustment
########################

def get_3dpos(pos, ob, r, t, K):
    """
    (3D)structure[point3d_id], (2D)key_points[j].pt, r, t, K
    """
    dtype = np.float32
    
    def F(x):
        p,J = cv2.projectPoints(x.reshape(1, 1, 3), r, t, K, np.array([]))
        p   = p.reshape(2)
        e   = ob - p
        err = e    
                
        return err
    res = least_squares(F, pos)
    return res.x

def get_3dpos_v1(pos,ob,r,t,K):
    """
    set the outliers to None, later delete them from the structure
    """
    p,J = cv2.projectPoints(pos.reshape(1, 1, 3), r, t, K, np.array([]))
    p = p.reshape(2)
    e = ob - p
    if abs(e[0]) > config.x or abs(e[1]) > config.y:        
        return None
    return pos

def bundle_adjustment(rotations, motions, K, correspond_struct_idx, key_points_for_all, structure):
    """
    optimize the 3D points and camera poses or just filter out the outliers, here for efficiency, we just filter out the outliers
    """
    for i in range(len(rotations)):
    # to make sure orthogonal, as well as less params
        r, _ = cv2.Rodrigues(rotations[i])
        rotations[i] = r

    for i in range(len(correspond_struct_idx)):
        point3d_ids = correspond_struct_idx[i]
        key_points  = key_points_for_all[i]
        r = rotations[i] 

        t = motions[i]
        for j in range(len(point3d_ids)):
            point3d_id = int(point3d_ids[j])
            if point3d_id < 0:
                continue
            new_point = get_3dpos_v1(structure[point3d_id], key_points[j].pt, r, t, K) #when using get_3dpos_v1, the structure will be filtered instead of being optimized
            
            structure[point3d_id] = new_point

    return structure

#######################
# plotting
#######################
    
def main(obj_name, device = 'cpu' ):

    imgdir    = os.path.join(config.image_dir,obj_name,'images','')
    # img_names = os.listdir()
    img_names = os.listdir(imgdir)
    img_names = sorted(img_names)
    # print(img_names[:5])
    # assert 1==0, "stop"
    
    for i in range(len(img_names)):
        img_names[i] = imgdir + img_names[i]

    # K is the intrinsic matrix
    K = config.K
    
    print('extracting the correspondences')
        
    key_points_for_all, descriptor_for_all, colors_for_all = extract_features(img_names, 'sift')
    matches_for_all = match_all_features(descriptor_for_all,device=device) #with shape(imgs-1,num_of_matches)
    assert len(matches_for_all) == len(img_names)-1, "Inconsistency of dimensions"

    print("initializing")
    structure, correspond_struct_idx, colors, rotations, motions = init_structure(K, key_points_for_all, colors_for_all, matches_for_all)   

    print("reconstruct")
    for i in range(1, len(matches_for_all)):

        object_points, image_points = get_objpoints_and_imgpoints(matches_for_all[i], correspond_struct_idx[i], structure, key_points_for_all[i + 1])

        # in cv2 len(fisrt param of solvePnPRansac) must > 7 
		# repeatedly padding the list using the first point until the length of the list is greater than 7
        if len(image_points) < 7:
            print(len(image_points))
            print(len(object_points))
            while len(image_points) < 7:
                object_points = np.append(object_points, [object_points[0]], axis = 0)
                image_points  = np.append(image_points, [image_points[0]], axis = 0)
   
        _, r, T, _ = cv2.solvePnPRansac(object_points, image_points, K, np.array([]))# get camera pose
        R, _       = cv2.Rodrigues(r) # transform r vector to R matrix

        rotations.append(R)
        motions.append(T)
        p1, p2 = get_matched_points(key_points_for_all[i], key_points_for_all[i + 1], matches_for_all[i])
        c1, c2 = get_matched_colors(colors_for_all[i], colors_for_all[i + 1], matches_for_all[i])
        # triangulation
        next_structure = reconstruct(K, rotations[i], motions[i], R, T, p1, p2)
        
        correspond_struct_idx[i], correspond_struct_idx[i + 1], structure, colors = fusion_structure(matches_for_all[i],correspond_struct_idx[i],correspond_struct_idx[i+1],structure,next_structure,colors,c1)
    
    print("filtering the outliers")
    structure = bundle_adjustment(rotations, motions, K, correspond_struct_idx, key_points_for_all, structure)
    
    i = 0
    # After BA, some points are filtered out, we need to delete them
    while i < len(structure):
        if math.isnan(structure[i][0]):
            structure = np.delete(structure, i, 0)
            colors    = np.delete(colors, i, 0)
            i -= 1
        i += 1
        
    print(len(structure))
    print(len(motions))

    print('done')
    return structure, colors, rotations, motions
if __name__ == '__main__':
    main()