import numpy as np

def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)

def cropped_spatial(a, focus, length):
    radius = int(length/2)
    tp = max(0, -(focus[0] - radius))
    bp = max(0, -((a.shape[1] - focus[0] - 1) - radius))
    lp = max(0, -(focus[1] - radius))
    rp = max(0, -((a.shape[2] - focus[1] - 1) - radius))
    
    # Create an empty array for the cropped image
    cropped_image = np.zeros((a.shape[0], 2 * radius + 1, 2 * radius + 1), dtype=a.dtype)
    cropped_image[:, tp: radius*2+1-bp, lp: radius*2+1-rp] = a[:,focus[0] - radius + tp:focus[0] + radius + bp +1, focus[1] - radius + lp :focus[1] +radius + rp +1]
    return cropped_image




