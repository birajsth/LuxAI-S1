import numpy as np

def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)

def cropped_spatial(a, focus, length):
    radius = int(length/2)
    tp = max(0, -(focus[0] - radius))
    bp = max(0, -((a.shape[1] - focus[0]) - radius))
    lp = max(0, -(focus[1] - radius))
    rp = max(0, -((a.shape[2] - focus[1]) - radius))

    return a[:,max(0,focus[0] - radius - bp):min(a.shape[1], focus[0] + radius + tp), \
            max(0,focus[1] - radius - rp):min(a.shape[2],focus[1] + radius + lp)]




