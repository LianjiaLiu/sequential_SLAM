#!/usr/bin/env python3
import torch


def match(descriptors1, descriptors2, device, dist="norm2", threshold=0, ratio=0.5):
    """
    Perform brute-force descriptor matching with Lowe's tests and cross-consistency check.

    Inputs:
    - descriptors1: Tensor of shape (N, feature_size) representing descriptors of keypoints.
    - descriptors2: Tensor of shape (M, feature_size) representing descriptors of keypoints.
    - device: Device where tensors will be allocated.
    - dist: Distance metric, "norm2" for Euclidean distance and "hamming" for Hamming distance.
    - threshold: Threshold for the first Lowe's test.
    - ratio: Ratio for the second Lowe's test.

    Returns:
    - matches: Tensor of shape (P, 2) containing indices of corresponding matches in the
      first and second sets of descriptors, where matches[:, 0] represent the indices
      in the first set and matches[:, 1] the indices in the second set of descriptors.
    """

    # Generate distance matrix 
    if dist == "hamming":
        p = 0
    else:
        p = 2.0

    distance_matrix = torch.cdist(descriptors1, descriptors2, p).to(device)

    matches = torch.zeros(0, 2).to(device)
    distances = torch.zeros(0).to(device)

    for idx_i, distance_matrix_i in enumerate(distance_matrix):
        # Sort distances and get top two distances and their indices
        sorted_distance_i_values, sorted_distance_i_indices = torch.topk(distance_matrix_i, k=2, largest=False)
        dist_i_top1, dist_i_top2 = sorted_distance_i_values
        idx_j_top1, idx_j_top2 = sorted_distance_i_indices
        
        # Perform Lowe's tests
        if dist_i_top1 <= threshold and dist_i_top1 <= dist_i_top2 * ratio:
            # Forward-backward consistency check
            sorted_distance_j_values, idx_i_top1 = torch.min(distance_matrix[:, idx_j_top1], dim=0)
            if idx_i_top1 == idx_i:
                # Append matches and distances
                matches = torch.cat((matches, torch.tensor([[idx_i_top1, idx_j_top1]], dtype=torch.long, device=device)), dim=0)
                distances = torch.cat((distances, torch.tensor([dist_i_top1], dtype=torch.float32, device=device)), dim=0)
                
    # Convert matches and distances to tensors
    matches = torch.tensor(matches, dtype=torch.long, device=device)
    distances = torch.tensor(distances, dtype=torch.float32, device=device)

    # Sort matches using distances from best to worst
    sorted_indices = distances.argsort()
    matches = matches[sorted_indices]

    return matches

