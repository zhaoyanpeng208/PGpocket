import numpy as np
import os
import yaml
import argparse
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import DataLoader
from torch_geometric import data as DATA
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from models.graph_gen import get_graph_generate_fn
from GCN_model_duoxc import GCNModel,PointGNNModel
from solver_duoxc import Solver
from read_file import PointCloudDataset

def main(rank,world_size):
    """CPU or GPU."""
    # create default process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')
    data_list = PointCloudDataset('./dataset/')
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data_list))
    train_idx = shuffled_indices[:int(0.8 * len(data_list))]
    val_idx = shuffled_indices[int(0.8 * len(data_list)):int(0.9 * len(data_list))]
    test_idx = shuffled_indices[int(0.9 * len(data_list)):]
    train_loader = DataLoader(data_list, batch_size=1, drop_last=True,
                              sampler=SubsetRandomSampler(train_idx.tolist()))
    val_loader = DataLoader(data_list, batch_size=1, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx.tolist()))
    test_loader = DataLoader(data_list, batch_size=1, drop_last=False,
                             sampler=SubsetRandomSampler(test_idx.tolist()))
    # Instantiate the model and other components
    model = PointGNNModel(in_features=16, hidden_features=300, num_classes=2,n_layers=4).to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    # model = PointNetModel().to(device)
    solver = Solver(ddp_model,optim=torch.optim.Adam,loss_func=nn.CrossEntropyLoss(), rank=rank, world_size=world_size)
    solver.train(train_loader, val_loader, eval_data=test_loader)

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29499"
    world_size = 2
    mp.spawn(main,
        args=(world_size,),
        nprocs=world_size,
        join=True)
