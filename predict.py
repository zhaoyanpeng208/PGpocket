import numpy as np
import os

from torch.utils.data import SubsetRandomSampler,BatchSampler,SequentialSampler
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
from util.config_util import save_config, save_train_config, \
    load_train_config, load_config
from GCN_model_duoxc import GCNModel,PointGNNModel
from solver_duoxc import Solver
from read_file import PointCloudDataset
from utils import get_true_site,get_pred_site,cal_dcc,get_pred_site_v2,write_pred_site
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, recall_score, precision_score, \
            accuracy_score, roc_auc_score, precision_recall_curve, auc, roc_curve

train_config = load_train_config('configs/car_auto_T3_train_train_config')
config_complete = load_config('configs/car_auto_T3_train_config')
config = config_complete

if torch.cuda.is_available():
    device = torch.device('cuda:1')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')


def main(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    data_list = PointCloudDataset('./dataset/holo4k_data/')
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data_list))
    train_idx = shuffled_indices[:int(0.8 * len(data_list))]
    val_idx = shuffled_indices[int(0.8 * len(data_list)):int(0.9 * len(data_list))]
    test_idx = shuffled_indices
    test_set = [data_list[i] for i in test_idx]
    train_loader = DataLoader(data_list, batch_size=1, drop_last=True,
                              sampler=SubsetRandomSampler(train_idx.tolist()))
    val_loader = DataLoader(data_list, batch_size=1, drop_last=False,
                            sampler=SubsetRandomSampler(val_idx.tolist()))
    test_loader = DataLoader(test_set, batch_size=1, drop_last=False,
                             sampler=SequentialSampler(test_set))
    # Instantiate the model and other components.
    # for j in range(1, 18):
        # checkpoint = torch.load(os.path.join('/home/g1/zyp/mymodel/runs/la_test1_08-11_08-35-13/checkpoint.pth'))
        # model = PointGNNModel(in_features=16, hidden_features=300, num_classes=2, n_layers=3).to(rank)
    checkpoint = torch.load(os.path.join('./runs/la_test1_07-12_16-56-38/' + str(15) + '_checkpoint.pth'))
    model = checkpoint.to(device)
    model.eval()
    # ddp_model = DDP(model, device_ids=[rank])
    # model = PointNetModel().to(device)
    # solver = Solver(model, optim=torch.optim.Adam, loss_func=nn.CrossEntropyLoss(), rank=rank,
    #                 world_size=world_size)
    #
    # # checkpoint = torch.load(os.path.join('/fs1/home/wangll/software/PointGNN/runs/la_test1_01-09_08-31-14/checkpoint.pth'))
    # solver.evaluation(test_loader, filename='test_data')
    p, y = [], []  # prediction and corresponding localization
    running_loss = 0
    sum_dcc = 0
    total_coord_list = torch.tensor([])
    pred_dic = {}

    for i, data in enumerate(test_loader):
        try:
            data = data.to(device)
            batch = data.batch
            label = data.y.to(device)
            batch_coord_list = torch.tensor([])
            for coord_list in data.vertex_coord_list:
                batch_coord_list = torch.cat((batch_coord_list, torch.tensor(coord_list[1])))
                total_coord_list = torch.cat((total_coord_list, torch.tensor(coord_list[1])))

            batch_coord_list = batch_coord_list.to(device)
            out = model(data.x, batch_coord_list, data.edge_index).to(device)
            out = out.tolist()
            label = label.tolist()
            out_np = np.array(out)
            out_np -= np.max(out_np, axis=1, keepdims=True)
            out_np = np.exp(out_np) / np.sum(np.exp(out_np), axis=1, keepdims=True)
            for l, o in enumerate(out):
                if o[0] >= o[1]:
                    pred = 0
                else:
                    pred = 1
                p.append(np.stack([pred, label[l]]))
            temp_p = np.squeeze(p)
            true_site_list = get_true_site(batch, temp_p[:, 1], batch_coord_list)
            pred_site_list = get_pred_site_v2(batch, out_np, batch_coord_list)
            total_center_coord_list = write_pred_site(batch, out_np, batch_coord_list)
            dcc = cal_dcc(true_site_list, pred_site_list)
            sum_dcc += dcc
            for i, pdb in enumerate(data.pdb):
                pred_dic[pdb] = [true_site_list[i], total_center_coord_list[i]]
                # np.save('./dataset/pred_dic.npy', pred_dic)

            # np.save('./dataset/pred_dic_vaild.npy',pred_dic)
            # np.save('./dataset/07-12_16-56-38/pred_dic_' + str(j) + '07-12_16-56-38.npy', pred_dic)
        except:
            print(data.pdb)

    np.save('./dataset/07-12_16-56-38/holo4k_15.npy', pred_dic)
    p = np.squeeze(p)
    acc = accuracy_score(p[:, 1], p[:, 0])
    f1 = f1_score(p[:, 1], p[:, 0])
    c_m = confusion_matrix(p[:, 1], p[:, 0])
    total_dcc = sum_dcc / (len(test_loader))
    running_loss /= len(test_loader)
    print(
        f'acc: {acc}, f1: {f1}')
    print(c_m)

    val_acc = 100 * np.equal(p[:, 0], p[:, 1]).sum() / len(p)
    val_mcc = matthews_corrcoef(p[:, 1], p[:, 0])
    val_f1 = f1_score(p[:, 1], p[:, 0])
    # val_auc = roc_auc_score(val_results[:, 1], prob_list)
    val_recall = recall_score(p[:, 1], p[:, 0])
    val_pre = precision_score(p[:, 1], p[:, 0])
    # fpr, tpr, tresholds = roc_curve(val_results[:, 1], prob_list, pos_label=1)
    # precision, recall, _thresholds = precision_recall_curve(val_results[:, 1], prob_list)
    # val_prauc = auc(recall, precision)
    print(val_acc, val_recall, val_pre, val_mcc, val_f1, total_dcc)

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29499"
    world_size = 1
    mp.spawn(main,
        args=(world_size,),
        nprocs=world_size,
        join=True)
    # main()
