import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric import data as DATA

from models.graph_gen import get_graph_generate_fn
from util.config_util import save_config, save_train_config, \
    load_train_config, load_config

train_config = load_train_config('configs/car_auto_T3_train_train_config')
config_complete = load_config('configs/car_auto_T3_train_config')
config = config_complete


class PointCloudDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PointCloudDataset,self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['dataset/Features.npy']

    @property
    def processed_file_names(self):
        return ['0.1_0.1_64_2.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = './dataset/holo4k_Features_npy.npy'
        pdb_list = './dataset/holo4k_data_ids.npy'
        protein_list = []
        loadData = np.load(data_list, allow_pickle=True)
        pdbdata = np.load(pdb_list, allow_pickle=True)
        for i, data in enumerate(loadData):
            labels = []
            point_features = []
            data_labels = data['labels']
            data_features = data['input_features']
            data_features = data_features.astype(np.float32)
            graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])  # 生成图'multi_level_local_graph_v3'
            (vertex_coord_list, keypoint_indices_list, edges_list) = \
                graph_generate_fn(data['xyz'], **config['graph_gen_kwargs'])
            # vertex_coord_list, keypoint_indices_list, edges_list = []
            for indice in keypoint_indices_list[0]:
                labels.append(data_labels[indice[0]][0])
                point_features.append(data_features[indice[0]])
            features = data_features

            vertex_coord_list = [p.astype(np.float32) for p in vertex_coord_list]
            keypoint_indices_list = [e.astype(np.int32) for e in keypoint_indices_list]
            edges_list = [e.astype(np.int32) for e in edges_list]

            protein_graph = DATA.Data(x = torch.tensor(point_features),
                                      edge_index = torch.LongTensor(edges_list[1]).transpose(1, 0),
                                      y = torch.LongTensor(labels))
            protein_graph.vertex_coord_list = vertex_coord_list
            protein_graph.keypoint_indices_list = keypoint_indices_list
            protein_graph.pdb = pdbdata[i]
            protein_graph.point_feature = features
            protein_graph.edges_list = edges_list
            print(i)
            protein_list.append(protein_graph)

        if self.pre_filter is not None:
            protein_list = [data for data in protein_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            protein_list = [self.pre_transform(data) for data in protein_list]

        data, slices = self.collate(protein_list)
        torch.save((data, slices), self.processed_paths[0])
