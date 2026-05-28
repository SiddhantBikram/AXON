import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 11
self_link = [(i, i) for i in range(num_node)]


inward_ori_index = [
(0, 2),   # Hip to mid(Hip,Neck)
(1, 2),   # mid(Hip,Neck) to Neck
(2, 3),   # Neck to Nose
(3, 4),   # Nose to Head
(2, 5),   # Neck to LShoulder
(5, 6),   # LShoulder to LElbow
(6, 7),   # LElbow to LWrist
(2, 8),   # Neck to RShoulder
(8, 9),   # RShoulder to RElbow
(9, 10),  # RElbow to RWrist
]

inward = [(i, j) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

num_node_1 = 11
indices_1 = [i for i in range(11)]
self_link_1 = [(i, i) for i in range(num_node_1)]
inward_ori_index_1 = [
(0, 2),   # Hip to mid(Hip,Neck)
(1, 2),   # mid(Hip,Neck) to Neck
(2, 3),   # Neck to Nose
(3, 4),   # Nose to Head
(2, 5),   # Neck to LShoulder
(5, 6),   # LShoulder to LElbow
(6, 7),   # LElbow to LWrist
(2, 8),   # Neck to RShoulder
(8, 9),   # RShoulder to RElbow
(9, 10),  # RElbow to RWrist
]
inward_1 =[(i, j) for (i, j) in inward_ori_index]
outward_1 = [(j, i) for (i, j) in inward]
neighbor_1 = inward_1 + outward_1

num_node_2 = 11
indices_2 = indices_1
self_link_2 = [(i ,i) for i in range(num_node_2)]
inward_ori_index_2 = [
(0, 2),   # Hip to mid(Hip,Neck)
(1, 2),   # mid(Hip,Neck) to Neck
(2, 3),   # Neck to Nose
(3, 4),   # Nose to Head
(2, 5),   # Neck to LShoulder
(5, 6),   # LShoulder to LElbow
(6, 7),   # LElbow to LWrist
(2, 8),   # Neck to RShoulder
(8, 9),   # RShoulder to RElbow
(9, 10),  # RElbow to RWrist
]
inward_2 = [(i, j) for (i, j) in inward_ori_index]
outward_2 = [(j, i) for (i, j) in inward]
neighbor_2 = inward_2 + outward_2

class Graph:
    def __init__(self, cfg, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.labeling_mode = labeling_mode #cfg.MODEL.LABELING_MODE
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A1 = tools.get_spatial_graph(num_node_1, self_link_1, inward_1, outward_1)
        self.A2 = tools.get_spatial_graph(num_node_2, self_link_2, inward_2, outward_2)
        self.A_binary = tools.edge2mat(neighbor, num_node)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2*np.eye(num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)

        self.A_A1 = ((self.A_binary + np.eye(num_node)) / np.sum(self.A_binary + np.eye(self.A_binary.shape[0]), axis=1, keepdims=True))[indices_1]
        self.A1_A2 = tools.edge2mat(neighbor_1, num_node_1) + np.eye(num_node_1)
        self.A1_A2 = (self.A1_A2 / np.sum(self.A1_A2, axis=1, keepdims=True))[indices_2]


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A