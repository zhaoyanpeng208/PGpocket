import os
from typing import Tuple
import time

import torch
import torch.nn as nn
import numpy as np
import warnings
from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score, recall_score, precision_score, \
            accuracy_score, roc_auc_score, precision_recall_curve, auc, roc_curve
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt
from utils import get_true_site,get_pred_site,cal_dcc,get_pred_site_v2,get_pred_site_v3
# from models.loss_functions import JointCrossEntropy
# from utils.general import tensorboard_confusion_matrix, padded_permuted_collate, plot_class_accuracies, \
#     tensorboard_class_accuracies, annotation_transfer, plot_confusion_matrix, LOCALIZATION


class Solver():
    def __init__(self, model, optim=torch.optim.Adam, loss_func=nn.CrossEntropyLoss(), eval=False, rank= None, world_size = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(rank)
        self.rank = rank
        self.world_size = world_size
        self.optim = optim(self.model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim,step_size=20,gamma=0.1)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=10)
        if eval:
            checkpoint = torch.load(os.path.join('', 'checkpoint.pt'),
                                    map_location=self.device)
            self.writer = SummaryWriter(
                'runs/{}_{}_{}'.format('la', 'test1',
                                       datetime.now().strftime('%d-%m_%H-%M-%S')))
            # with open(os.path.join(self.writer.log_dir, 'epoch.txt'), "r") as f:  # last epoch not the best epoch
            #     self.start_epoch = int(f.read()) + 1
            self.max_val_acc = checkpoint['maximum_accuracy']
            # self.weight = checkpoint['weight'].to(self.device)

        if not eval:
            self.start_epoch = 0
            self.max_val_f1 = 0  # running accuracy to decide whether or not a new model should be saved
            self.max_dcc = 0
            self.writer = SummaryWriter(
                'runs/{}_{}_{}'.format('la', 'test1',
                                       datetime.now().strftime('%d-%m_%H-%M-%S')))
            # self.weight = weight.to(self.device)

        weights = [1, 3]
        class_weights = torch.FloatTensor(weights).to(self.rank)
        self.loss_func = nn.CrossEntropyLoss(weight=class_weights)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, eval_data=None):
        """
        Train and simultaneously evaluate on the val_loader and then estimate the stderr on eval_data if it is provided
        Args:
            train_loader: For training
            val_loader: For validation during training
            eval_data: For evaluation and estimating stderr after training

        Returns:

        """
        epochs_no_improve = 0  # counts every epoch that the validation accuracy did not improve for early stopping
        min_train_f1 = 0
        max_train_f1 = 0
        for epoch in range(self.start_epoch, 300):  # loop over the dataset multiple times
            self.model.train()
            start_time = time.time()
            train_loss, train_results = self.predict(train_loader, epoch + 1, optim=self.optim)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"训练运行时间: {elapsed_time} 秒")

            self.model.eval()
            start_time = time.time()
            with torch.no_grad():
                val_loss, val_results, coord_list, dcc = self.predict(val_loader, optim=None)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"验证运行时间: {elapsed_time} 秒")

            val_results = np.array(val_results)
            val_results = np.squeeze(val_results)
            train_acc = 100 * np.equal(train_results[:, 0], train_results[:, 1]).sum() / len(train_results)
            val_acc = 100 * np.equal(val_results[:, 0], val_results[:, 1]).sum() / len(val_results)
            with warnings.catch_warnings():  # because sklearns mcc implementation is a little dim
                warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
                train_mcc = matthews_corrcoef(train_results[:, 1], train_results[:, 0])
                val_mcc = matthews_corrcoef(val_results[:, 1], val_results[:, 0])
                train_f1 = f1_score(train_results[:, 1], train_results[:, 0])
                val_f1 = f1_score(val_results[:, 1], val_results[:, 0])

            print('[Epoch %d] val accuracy: %.4f%% train accuracy: %.4f%% dcc: %.4f%%' % (epoch + 1, val_acc, train_acc, dcc))

            self.writer.add_scalars('Acc', {'train': train_acc, 'val': val_acc}, epoch + 1)
            self.writer.add_scalars('MCC', {'train': train_mcc, 'val': val_mcc}, epoch + 1)
            self.writer.add_scalars('F1', {'train': train_f1, 'val': val_f1}, epoch + 1)
            self.writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch + 1)

            # print(c)
            # print(var)
            # self.writer.add_scalars('Loss', {'train': train_loss, 'val': np.array(c)}, epoch + 1)
            # self.writer.add_scalars('confidence', {'c': np.array(c), 'val': np.array(var)}, epoch + 1)

            if val_f1 >= self.max_val_f1:  # save the model with the best accuracy
                epochs_no_improve = 0
                self.max_val_f1 = val_f1

            else:
                epochs_no_improve += 1
            self.save_checkpoint(epoch + 1)
            with open(os.path.join(self.writer.log_dir, 'epoch.txt'), 'w') as file:  # save what the last epoch is
                file.write(str(epoch))

            if train_f1 >= max_train_f1:
                max_train_f1 = train_f1
            if epochs_no_improve >= 20 and max_train_f1 >= min_train_f1:  # stopping criterion
                break

        if eval_data:  # do evaluation on the test data if a eval_data is provided
            # load checkpoint of best model to do evaluation
            checkpoint = torch.load(os.path.join(self.writer.log_dir, 'checkpoint.pth'))
            self.model = checkpoint
            self.evaluation(eval_data, filename='val_data_after_training')

    # train
    def predict(self, data_loader: DataLoader, epoch: int = None, optim: torch.optim.Optimizer = None):
        """
        get predictions for data in dataloader and do backpropagation if an optimizer is provided
        Args:
            data_loader: pytorch dataloader from which the batches will be taken
            epoch: optional parameter for logging
            optim: pytorch optimiz. If this is none, no backpropagation is done

        Returns:
            loc_loss: the average of the localization loss accross all batches
            sol_loss: the average of the solubility loss across all batches
            results: localizations # [n_train_proteins, 2] predictions in first and loc in second position
        """
        p, y = [],[]  # prediction and corresponding localization
        running_loss = 0
        sum_dcc = 0
        total_coord_list = torch.tensor([])
        total_pred_dic = {}
        for i, data in enumerate(data_loader):
            data = data.to(self.rank)
            batch = data.batch
            label = data.y.to(self.rank)
            batch_coord_list = torch.tensor([])
            for coord_list in data.vertex_coord_list:
                batch_coord_list = torch.cat((batch_coord_list, torch.tensor(coord_list[1])))
                total_coord_list = torch.cat((total_coord_list,torch.tensor(coord_list[1])))

            batch_coord_list = batch_coord_list.to(self.rank)
            out = self.model(data.x, batch_coord_list, data.edge_index).to(self.rank)

            loss = self.loss_func(out, label)
            # loss = loss * class_weights * mask
            loss = loss.sum()
            if optim:  # run backpropagation if an optimizer is provided
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

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
            if epoch:
                running_loss += loss
            else:
                running_loss += loss
                # true_site_list = get_true_site(batch, temp_p[:, 1], batch_coord_list)
                pred_site_list = get_pred_site_v3(batch, out_np, batch_coord_list)
                total_pred_dic[data.pdb[0]] = pred_site_list
                # dcc = cal_dcc(true_site_list,pred_site_list)
                # sum_dcc+=dcc


            if i % 100 == 99:  # log every log_iterations
                if epoch:
                    print('Epoch %d ' % (epoch), end=' ')
                    print('[Iter %5d/%5d] %s: loss: %.7f' % (
                        i + 1, len(data_loader), 'Train' if optim else 'Val', loss))
        p = np.squeeze(p)
        acc = accuracy_score(p[:, 1], p[:, 0])
        f1 = f1_score(p[:, 1], p[:, 0])
        c_m = confusion_matrix(p[:, 1], p[:, 0])
        dis_pdb_file = './dataset/test_pdb/true_site_dis_cal_dic.npy'
        dis_dic = np.load(dis_pdb_file, allow_pickle=True).item()
        sum_dcc = cal_dcc(dis_dic,total_pred_dic)
        total_dcc = sum_dcc/(len(data_loader))
        running_loss /= len(data_loader)
        print(
            f'acc: {acc}, f1: {f1}')
        print(c_m)
        if epoch:
            return running_loss, p
        else:
            return running_loss, p, total_coord_list, total_dcc


    def evaluation(self, eval_dataset, filename: str = '', lookup_dataset: Dataset = None):

        self.model.eval()
        with torch.no_grad():
            val_loss, val_results, coord_list, dcc = self.predict(eval_dataset)
        # print(p,c,var)
        val_acc = 100 * np.equal(val_results[:, 0], val_results[:, 1]).sum() / len(val_results)
        val_mcc = matthews_corrcoef(val_results[:, 1], val_results[:, 0])
        val_f1 = f1_score(val_results[:, 1], val_results[:, 0])
        # val_auc = roc_auc_score(val_results[:, 1], prob_list)
        val_recall = recall_score(val_results[:, 1], val_results[:, 0])
        val_pre = precision_score(val_results[:, 1], val_results[:, 0])
        # fpr, tpr, tresholds = roc_curve(val_results[:, 1], prob_list, pos_label=1)
        # precision, recall, _thresholds = precision_recall_curve(val_results[:, 1], prob_list)
        # val_prauc = auc(recall, precision)
        print(val_acc, val_recall, val_pre, val_mcc, val_f1,dcc)
        matrixs = [val_acc, val_recall, val_pre, val_mcc, val_f1,dcc]
        with open('matrix_result.csv', 'a') as f:
            f.write('\t'.join(map(str, matrixs)) + '\n')


    def save_checkpoint(self, epoch: int):
        """
        Saves checkpoint of model in the logdir of the summarywriter/ in the used rundir
        Args:
            epoch: current epoch from which the run will be continued if it is loaded

        Returns:

        """
        run_dir = self.writer.log_dir
        torch.save(self.model, os.path.join(run_dir,str(epoch)+ '_checkpoint.pth'))


