import os
import torch
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans
import numpy as np
from numpy import where
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import OPTICS

def del_files():
    # 指定要删除文件名的文件夹路径
    folder_path = '/data0/zyp/mymodel/dataset/01-benchmark_pdbs'

    # 打开包含文件名的文本文件
    with open('/data0/zyp/mymodel/dataset/bug_list.txt', 'r') as file:
        # 读取文件中的文件名列表
        file_list = file.read().splitlines()

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 如果文件名在文件名列表中，就删除该文件
        if filename in file_list:
            file_path = os.path.join(folder_path, filename)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

def get_true_site(batch, label, batch_coord_list):
    # 获取真实标签的中心坐标
    center_coord_list = []
    true_site_list = []
    for k, b in enumerate(batch):
        if b != batch[-1]:
            # 同一batch下且标签为1
            if batch[k] == batch[k + 1] and label[k] == 1:
                true_site_list.append(batch_coord_list[k])
            # batch交界处
            elif batch[k] != batch[k + 1]:
                if true_site_list:
                    true_site_tensor = torch.stack(true_site_list, dim=1)
                    center_coord = torch.mean(true_site_tensor, dim=1)
                    center_coord_list.append(center_coord)
                else:
                    center_coord = torch.tensor([0,0,0])
                    center_coord_list.append(center_coord)
                true_site_list = []
        # batch中最后一个样本
        elif b == batch[-1]:
            if label[k] == 1:
                true_site_list.append(batch_coord_list[k])
            if k == len(batch) - 1:
                if true_site_list:
                    true_site_tensor = torch.stack(true_site_list, dim=1)
                    center_coord = torch.mean(true_site_tensor, dim=1)
                    center_coord_list.append(center_coord)
                else:
                    center_coord = torch.tensor([0, 0, 0])
                    center_coord_list.append(center_coord)
    return center_coord_list, true_site_list


def get_pred_site(batch, label, batch_coord_list):
    '''
    获取预测标签的中心坐标
    对所有预测结果进行排名，只取排名最高的
    :param batch:
    :param label:
    :param batch_coord_list:
    :return:
    '''
    top_center_coord_list = []
    true_site_list = []
    for k, b in enumerate(batch):
        if b != batch[-1]:
            # 同一batch下且标签为1
            if batch[k] == batch[k + 1] and label[k][0]<label[k][1]:
                true_site_list.append([batch_coord_list[k].cpu().numpy(), label[k][1]])
            # batch交界处
            elif batch[k] != batch[k + 1]:
                pred_site_coord = []
                pred_site_p = []
                center_coord_list = []
                # center_coord_list.append(batch[k])
                for s in true_site_list:
                    pred_site_coord.append(s[0])
                    pred_site_p.append(s[1])
                pred_site_coord =np.array(pred_site_coord)
                # print(pred_site_coord)
                if pred_site_coord.any():
                    for ind,_ in enumerate(pred_site_coord):
                        center_coord_list.append([pred_site_coord[ind], pred_site_p[ind]])
                    top_coord = sorted(center_coord_list, key=lambda x: x[1],reverse=True)[0]
                    top_center_coord_list.append(top_coord)
                else:
                    top_center_coord_list.append([0, 0, 0])
                true_site_list = []
        # batch中最后一个样本
        elif b == batch[-1]:
            if label[k][0]<label[k][1]:
                true_site_list.append([batch_coord_list[k].cpu().numpy(), label[k][1]])
            if k == len(batch) - 1:
                center_coord_list = []
                pred_site_coord = []
                pred_site_p = []
                # center_coord_list.append(batch[k])
                for s in true_site_list:
                    pred_site_coord.append(s[0])
                    pred_site_p.append(s[1])
                pred_site_coord = np.array(pred_site_coord)
                # print(pred_site_coord)
                if pred_site_coord.any():
                    for ind, _ in enumerate(pred_site_coord):
                        center_coord_list.append([pred_site_coord[ind], pred_site_p[ind]])
                    top_coord = sorted(center_coord_list, key=lambda x: x[1],reverse=True)[0]
                    top_center_coord_list.append(top_coord)
                else:
                    top_center_coord_list.append(np.array([0,0,0]))
    return top_center_coord_list


def get_pred_site_temp(batch, label, batch_coord_list):
    '''
    首先对所有预测位点进行AP聚类，对聚类中心进行排名，只取排名最高的
    :param batch:
    :param label:
    :param batch_coord_list:
    :return:
    '''
    #获取预测标签的中心坐标
    top_center_coord_list = []
    true_site_list = []
    for k, b in enumerate(batch):
        if b != batch[-1]:
            # 同一batch下且标签为1
            if batch[k] == batch[k + 1] and label[k][0]<label[k][1]:
                true_site_list.append([batch_coord_list[k].cpu().numpy(), label[k][1]])
            # batch交界处
            elif batch[k] != batch[k + 1]:
                pred_site_coord = []
                pred_site_p = []
                center_coord_list = []
                # center_coord_list.append(batch[k])
                for s in true_site_list:
                    pred_site_coord.append(s[0])
                    pred_site_p.append(s[1])
                pred_site_coord =np.array(pred_site_coord)
                # print(pred_site_coord)
                if pred_site_coord.any():
                    center_coord_indices = AP_cluster(pred_site_coord, label)
                    for ind in center_coord_indices:
                        center_coord_list.append([pred_site_coord[ind], pred_site_p[ind]])
                    top_coord = sorted(center_coord_list, key=lambda x: x[1])[0]
                    top_center_coord_list.append(top_coord)
                else:
                    top_center_coord_list.append([0, 0, 0])
                true_site_list = []
        # batch中最后一个样本
        elif b == batch[-1]:
            if label[k][0]<label[k][1]:
                true_site_list.append([batch_coord_list[k].cpu().numpy(), label[k][1]])
            if k == len(batch) - 1:
                center_coord_list = []
                pred_site_coord = []
                pred_site_p = []
                # center_coord_list.append(batch[k])
                for s in true_site_list:
                    pred_site_coord.append(s[0])
                    pred_site_p.append(s[1])
                pred_site_coord = np.array(pred_site_coord)
                # print(pred_site_coord)
                if pred_site_coord.any():
                    center_coord_indices = AP_cluster(pred_site_coord, label)
                    for ind in center_coord_indices:
                        center_coord_list.append([pred_site_coord[ind], pred_site_p[ind]])
                    top_coord = sorted(center_coord_list, key=lambda x: x[1])[0]
                    top_center_coord_list.append(top_coord)
                else:
                    top_center_coord_list.append(np.array([0,0,0]))
    return top_center_coord_list

def get_pred_site_v2(batch, label, batch_coord_list):
    '''
    首先对所有预测位点进行AP聚类，对聚类中心进行排名，只取排名最高的
    :param batch:
    :param label:
    :param batch_coord_list:
    :return:
    '''
    #获取预测标签的中心坐标
    top_center_coord_list = []
    true_site_list = []
    for k, b in enumerate(batch):
        if b != batch[-1]:
            # 同一batch下且标签为1
            if batch[k] == batch[k + 1] and label[k][0]<label[k][1]:
                true_site_list.append([batch_coord_list[k].cpu().numpy(), label[k][1]])
            # batch交界处
            elif batch[k] != batch[k + 1]:
                pred_site_coord = []
                pred_site_p = []
                center_coord_list = []
                # center_coord_list.append(batch[k])
                for s in true_site_list:
                    pred_site_coord.append(s[0])
                    pred_site_p.append(s[1])
                pred_site_coord =np.array(pred_site_coord)
                # print(pred_site_coord)
                if pred_site_coord.any():
                    center_coord_model = AP_cluster(pred_site_coord, label)
                    center_coord_lists = split_list_by_index(pred_site_coord, center_coord_model.labels_,pred_site_p)
                    for l in center_coord_lists:
                        coord = [sublist[0] for sublist in l]
                        p = [sublist[1] for sublist in l]
                        ave_coord = np.mean(coord, axis=0)
                        ave_p = np.mean(p)
                        center_coord_list.append([ave_coord,ave_p])
                    top_coord = sorted(center_coord_list, key=lambda x: x[1],reverse=True)[0]
                    top_center_coord_list.append(top_coord)
                else:
                    top_center_coord_list.append([0, 0, 0])
                true_site_list = []
        # batch中最后一个样本
        elif b == batch[-1]:
            if label[k][0]<label[k][1]:
                true_site_list.append([batch_coord_list[k].cpu().numpy(), label[k][1]])
            if k == len(batch) - 1:
                pred_site_coord = []
                pred_site_p = []
                center_coord_list = []
                # center_coord_list.append(batch[k])
                for s in true_site_list:
                    pred_site_coord.append(s[0])
                    pred_site_p.append(s[1])
                pred_site_coord = np.array(pred_site_coord)
                # print(pred_site_coord)
                if pred_site_coord.any():
                    center_coord_model = AP_cluster(pred_site_coord, label)
                    center_coord_lists = split_list_by_index(pred_site_coord, center_coord_model.labels_, pred_site_p)
                    for l in center_coord_lists:
                        coord = [sublist[0] for sublist in l]
                        p = [sublist[1] for sublist in l]
                        ave_coord = np.mean(coord, axis=0)
                        ave_p = np.mean(p)
                        center_coord_list.append([ave_coord, ave_p])
                    top_coord = sorted(center_coord_list, key=lambda x: x[1], reverse=True)[0]
                    top_center_coord_list.append(top_coord)
                else:
                    top_center_coord_list.append(np.array([0,0,0]))
    return top_center_coord_list

def get_pred_site_v3(batch, label, batch_coord_list):
    '''
    首先对所有预测位点进行OPTICS聚类，对聚类中心进行排名，只取排名最高的
    :param batch:
    :param label:
    :param batch_coord_list:
    :return:
    '''
    #获取预测标签的中心坐标
    top_center_coord_list = []
    true_site_list = []
    for k, b in enumerate(batch):
        if b != batch[-1]:
            # 同一batch下且标签为1
            if batch[k] == batch[k + 1] and label[k][0]<label[k][1]:
                true_site_list.append([batch_coord_list[k].cpu().numpy(), label[k][1]])
            # batch交界处
            elif batch[k] != batch[k + 1]:
                clu_res = {}
                pred_site_coord = []
                pred_site_p = []
                center_coord_list = []
                # center_coord_list.append(batch[k])
                for s in true_site_list:
                    pred_site_coord.append(s[0])
                    pred_site_p.append(s[1])
                pred_site_coord =np.array(pred_site_coord)
                # print(pred_site_coord)
                if pred_site_coord.any():
                    center_coord_model = AP_cluster(pred_site_coord, label)
                    center_coord_model.fit(pred_site_coord)
                    labels = center_coord_model.labels_
                    for label, feature in zip(labels, pred_site_coord):
                        if label not in clu_res:
                            clu_res[label] = [feature]
                        else:
                            clu_res[label].append(feature)
                    top_coord = sorted(center_coord_list, key=lambda x: x[1],reverse=True)[0]
                    top_center_coord_list.append(top_coord)
                else:
                    top_center_coord_list.append([0, 0, 0])
                true_site_list = []
        # batch中最后一个样本
        elif b == batch[-1]:
            if label[k][0]<label[k][1]:
                true_site_list.append([batch_coord_list[k].cpu().numpy(), label[k][1]])
            if k == len(batch) - 1:
                distance = 100
                min_len = 5
                clu_res = {}
                pred_site_coord = []
                pred_site_p = []
                center_coord_list = []
                # center_coord_list.append(batch[k])
                for s in true_site_list:
                    pred_site_coord.append(s[0])
                    pred_site_p.append(s[1])
                pred_site_coord = np.array(pred_site_coord)
                # print(pred_site_coord)
                if pred_site_coord.any():
                    try:
                        center_coord_model = OPTICS(min_samples=2, xi=0.5, min_cluster_size=0.3)
                        center_coord_model.fit(pred_site_coord)
                        labels = center_coord_model.labels_
                        for label, feature in zip(labels, pred_site_coord):
                            if label not in clu_res:
                                clu_res[label] = [feature]
                            else:
                                clu_res[label].append(feature)
                        for clu in clu_res:
                            if clu != -1:
                                if len(clu_res[clu]) > min_len:
                                    min_len = len(clu_res[clu])
                                    mean_point = np.mean(clu_res[clu], axis=0)
                                    # mean_point = cal_mean_points(clu_res[clu])
                                    top_center_coord_list.append(mean_point)
                    except:
                        mean_point = np.mean(pred_site_coord, axis=0)
                        top_center_coord_list.append(mean_point)
                else:
                    top_center_coord_list.append(np.array([0,0,0]))
    return top_center_coord_list


def AP_cluster(coordinate,coordinate_line):
    # （1）亲和力传播算法AP
    X = coordinate
    # X1 = coordinate
    X = StandardScaler().fit_transform(X)
    # 定义模型
    model = AffinityPropagation(damping=0.5)  # 设置damping : 阻尼系数，取值[0.5,1)
    # 匹配模型
    model.fit(X)

    # return model.cluster_centers_indices_
    return model

# def cal_dcc(true_site_list,pred_site_list):
#     number_true = 0
#     for i,site in enumerate(true_site_list):
#         dis = np.linalg.norm(true_site_list[i].cpu().numpy() - pred_site_list[i][0])
#         if dis<=4:
#             number_true += 1
#     return number_true

def split_list_by_index(data_list, index_list, label_list):
    max_index = max(index_list)
    result_lists = [[] for _ in range(max_index + 1)]

    for i, data in enumerate(data_list):
        index = index_list[i]
        result_lists[index].append([data,label_list[i]])

    return result_lists

def write_pred_site(batch, label, batch_coord_list):
    '''
    记录正样本
    '''
    top_center_coord_list = []
    true_site_list = []
    total_site_list = []
    total_center_coord_list = []
    for k, b in enumerate(batch):
        if b != batch[-1]:
            # 同一batch下且标签为1
            if batch[k] == batch[k + 1] and label[k][0]<label[k][1]:
                true_site_list.append([batch_coord_list[k].cpu().numpy(), label[k][1]])
            # batch交界处
            elif batch[k] != batch[k + 1]:
                pred_site_coord = []
                pred_site_p = []
                center_coord_list = []
                # center_coord_list.append(batch[k])
                for s in true_site_list:
                    pred_site_coord.append(s[0])
                    pred_site_p.append(s[1])
                pred_site_coord =np.array(pred_site_coord)
                # print(pred_site_coord)
                if pred_site_coord.any():
                    center_coord_model = AP_cluster(pred_site_coord, label)
                    center_coord_lists = split_list_by_index(pred_site_coord, center_coord_model.labels_,pred_site_p)
                    for l in center_coord_lists:
                        coord = [sublist[0] for sublist in l]
                        p = [sublist[1] for sublist in l]
                        ave_coord = np.mean(coord, axis=0)
                        ave_p = np.mean(p)
                        center_coord_list.append([ave_coord,ave_p])
                    top_coord = sorted(center_coord_list, key=lambda x: x[1],reverse=True)[0]
                    top_center_coord_list.append(top_coord)
                    total_center_coord_list.append(center_coord_list)
                else:
                    top_center_coord_list.append(np.array([0,0,0]))
                    total_center_coord_list.append(np.array([0,0,0]))
                total_site_list.append(true_site_list)
                true_site_list = []

        # batch中最后一个样本
        elif b == batch[-1]:
            if label[k][0]<label[k][1]:
                true_site_list.append([batch_coord_list[k].cpu().numpy(), label[k][1]])
            if k == len(batch) - 1:
                try:
                    pred_site_coord = []
                    pred_site_p = []
                    center_coord_list = []
                    # center_coord_list.append(batch[k])
                    for s in true_site_list:
                        pred_site_coord.append(s[0])
                        pred_site_p.append(s[1])
                    pred_site_coord = np.array(pred_site_coord)
                    # print(pred_site_coord)
                    if pred_site_coord.any():
                        center_coord_model = AP_cluster(pred_site_coord, label)
                        center_coord_lists = split_list_by_index(pred_site_coord, center_coord_model.labels_, pred_site_p)
                        for l in center_coord_lists:
                            coord = [sublist[0] for sublist in l]
                            p = [sublist[1] for sublist in l]
                            ave_coord = np.mean(coord, axis=0)
                            ave_p = np.mean(p)
                            center_coord_list.append([ave_coord, ave_p])
                        top_coord = sorted(center_coord_list, key=lambda x: x[1], reverse=True)[0]
                        top_center_coord_list.append(top_coord)
                        total_center_coord_list.append(center_coord_list)
                    else:
                        top_center_coord_list.append(np.array([0,0,0]))
                        total_center_coord_list.append(np.array([0,0,0]))
                    total_site_list.append(true_site_list)
                except:
                    top_center_coord_list.append(np.array([0, 0, 0]))
                    total_center_coord_list.append(np.array([0, 0, 0]))
    return total_site_list

import os
import gzip
from tqdm import tqdm

def unzip_pdb_file(source_files_folder, target_files_folder):
    """
    解压各文件夹下的.gz文件
    :param source_files_folder:
    :param target_files_folder:
    :return:
    """
    # 遍历源文件夹及其子文件夹
    for root, dirs, files in tqdm(os.walk(source_files_folder, target_files_folder)):
        for file in files:
            # 检查文件是否为ZIP文件
            try:
                if file.endswith('.gz'):
                    pdb_name = file.split('.')[0]
                    # 构建ZIP文件的完整路径
                    zip_file_path = os.path.join(root, file)

                    # 创建与ZIP文件相同的目标文件夹结构
                    relative_path = os.path.relpath(zip_file_path, source_files_folder)
                    extract_folder = os.path.join(target_files_folder, os.path.dirname(relative_path))
                    os.makedirs(extract_folder, exist_ok=True)

                    # 解压ZIP文件到目标文件夹
                    g_file = gzip.GzipFile(zip_file_path)
                    open(extract_folder+'/'+pdb_name+'.ent', "wb+").write(g_file.read())
                    g_file.close()
            except:
                print(file)

# source_files_folder = './dataset/test_pdb/raw/'
# target_files_folder = './dataset/test_pdb/process/'
# unzip_pdb_file(source_files_folder,target_files_folder)

import os
from Bio import PDB


class ChainSplitter:
    def __init__(self, out_dir=None):
        """ Create parsing and writing objects, specify output directory."""
        self.parser = PDB.PDBParser()
        self.writer = PDB.PDBIO()
        if out_dir is None:
            out_dir = os.path.join(os.getcwd(),"chain_PDBs")
        self.out_dir = out_dir

    def make_pdb(self, pdb_path, chain_letters, overwrite=False, struct=None):
        """ Create a new PDB file containing only the specified chains.
        Returns the path to the created file.
        :param pdb_path: full path to the crystal structure
        :param chain_letters: iterable of chain characters (case insensitive)
        :param overwrite: write over the output file if it exists
        """
        chain_letters = [chain.upper() for chain in chain_letters]

        # Input/output files
        (pdb_dir, pdb_fn) = os.path.split(pdb_path)
        pdb_id = pdb_fn[0:4]
        out_name ="pdb%s_%s.ent" % (pdb_id,"".join(chain_letters))
        out_path = os.path.join(self.out_dir, out_name)
        print("OUT PATH:",out_path)
        plural ="s" if (len(chain_letters) > 1) else""  # for printing

        # Skip PDB generation if the file already exists
        if (not overwrite) and (os.path.isfile(out_path)):
            print("Chain%s %s of '%s' already extracted to '%s'." %
                    (plural,",".join(chain_letters), pdb_id, out_name))
            return out_path

        print("Extracting chain%s %s from %s..." % (plural,
               ",".join(chain_letters), pdb_fn))

        # Get structure, write new file with only given chains
        if struct is None:
            struct = self.parser.get_structure(pdb_id, pdb_path)
        self.writer.set_structure(struct)
        self.writer.save(out_path, select=SelectChains(chain_letters))

        return out_path


class SelectChains(PDB.Select):
    """ Only accept the specified chains when saving."""
    def __init__(self, chain_letters):
        self.chain_letters = chain_letters

    def accept_chain(self, chain):
        return (chain.get_id() in self.chain_letters)

import sys
def get_pdb_chain():
    """ Parses PDB id's desired chains, and creates new PDB structures."""
    file_path = './dataset/pred_dic.npy'
    input_pdb_file = './dataset/test_pdb/process/'
    output_pdb_file = './dataset/test_pdb/processed/'
    pred_dic = np.load(file_path, allow_pickle=True).item()
    splitter = ChainSplitter("/home/g1/zyp/mymodel/dataset/test_pdb/processed")
    for pdb in pred_dic:
        pdb_id = pdb.split('_')[0]
        chain = pdb.split('_')[1]
        # pdb_fn = pdbList.retrieve_pdb_file(pdb_id)
        splitter.make_pdb(input_pdb_file+pdb_id+'.ent', chain)

def cal_dcc(dis_dic, pred_dic):
    dcc = 0
    for pdb in pred_dic:
        try:
            distance = 100
            # if pdb == '4guk_B':
                # print(pdb)
            clu_res = {}
            true_site = dis_dic[pdb][0]
            pred_sites_info = pred_dic[pdb]
            distance = np.linalg.norm(pred_sites_info - true_site)
            if distance<=4:
                dcc+=1
            # print(distance)
        except:
            # print(pdb)
            continue
    return dcc


import os
import shutil

def compare_and_copy(folder_a, folder_b, folder_c):
    '''
    请将 /path/to/folder_a、/path/to/folder_b 和 /path/to/folder_c 替换为实际的文件夹路径。
    这个脚本将找到在 b 文件夹中存在但在 a 文件夹中不存在的文件，并将它们复制到 c 文件夹中。
    :param folder_a:
    :param folder_b:
    :param folder_c:
    :return:
    '''
    files_a = set(os.listdir(folder_a))
    files_b = set(os.listdir(folder_b))
    new_files_a = []
    for file in files_a:
        temp_file = file[0:4]+file[6:]
        new_files_a.append(temp_file)
    new_files_a = set(new_files_a)
    files_only_in_b = files_b - new_files_a

    # 创建文件夹 c（如果不存在）
    os.makedirs(folder_c, exist_ok=True)

    # 将文件从 b 复制到 c
    for file_name in files_only_in_b:
        source_path = os.path.join(folder_b, file_name)
        destination_path = os.path.join(folder_c, file_name)
        shutil.copy2(source_path, destination_path)

# # 示例用法
# folder_a_path = '/home/g1/zyp/mymodel/dataset/01-benchmark_pdbs/'
# folder_b_path = '/home/g1/zyp/mymodel/dataset/holo4k/'
# folder_c_path = '/home/g1/zyp/mymodel/dataset/holo4k_new/'
#
# compare_and_copy(folder_a_path, folder_b_path, folder_c_path)
# print("比较结果已保存到", folder_c_path)

import os
import csv
from Bio import PDB

def extract_chain_and_ligand_info_for_folder(folder_path, output_csv_path):
    '''
    这个脚本将链和配体信息写入一个 CSV 文件中，其中每一行包含文件名、模型、链和残基信息。
    :param folder_path:
    :param output_csv_path:
    :return:
    '''
    # 创建输出文件夹（如果不存在）
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # 创建 CSV 文件并写入标题行
    with open(output_csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['File', 'Model', 'Chain', 'Residue'])

        for file_name in os.listdir(folder_path):
            if file_name.endswith('.pdb'):
                pdb_file_path = os.path.join(folder_path, file_name)
                rows = extract_chain_and_ligand_info(pdb_file_path, file_name)

                # 写入 CSV 文件
                csv_writer.writerows(rows)

                print(f"处理完成: {pdb_file_path}")

def extract_chain_and_ligand_info(pdb_file_path, file_name):
    structure = PDB.PDBParser(QUIET=True).get_structure('structure', pdb_file_path)

    rows = []

    for model in structure:
        for chain in model:
            # rows.append([file_name, model.id, chain.id, ''])

            for residue in chain:
                if PDB.is_aa(residue):
                    continue  # 跳过氨基酸，只关注配体
                if residue.id[0][0] == 'H':
                    rows.append([file_name, model.id, chain.id, residue.id])

    return rows

# # 示例用法
# pdb_folder_path = '/home/g1/zyp/mymodel/dataset/holo4k_new'
# output_csv_path = '/home/g1/zyp/mymodel/dataset/holo4k_info.csv'
#
# extract_chain_and_ligand_info_for_folder(pdb_folder_path, output_csv_path)

import pandas as pd
def remove_others_ligand():
    # 读取 CSV 文件
    csv_file_path = '/home/g1/zyp/mymodel/dataset/holo4k_info.csv'
    df = pd.read_csv(csv_file_path).values
    new_df = []
    # # 删除包含指定字符串的行
    strings_to_remove = ['ZN', 'CL', 'MN', 'MG', 'CA', 'SO4', 'NA', 'PO4','NAG','GOL']
    # df_filtered = df[~df['r'].str.contains('|'.join(strings_to_remove))]

    for line in df:
        lig_info = line[3][4:7]
        if lig_info.find("'")!=-1 or lig_info in strings_to_remove:
            continue
        new_df.append(line)

    # 保存结果到新的 CSV 文件
    output_file_path = '/home/g1/zyp/mymodel/dataset/holo4k_data/holo4k_info_new.csv'
    df_filtered = pd.DataFrame(new_df)
    df_filtered.to_csv(output_file_path, index=False)

    print(f"已保存过滤后的结果到 {output_file_path}")
# remove_others_ligand()

def remove_chains(input_csv_path, output_csv_path):
    # 读取 CSV 文件
    df = pd.read_csv(input_csv_path)

    # 对相同 pdb 值的行进行分组，对不同 chain 对应的 lig 进行比较
    grouped_df = df.groupby('pdb')

    # 保存结果的列表
    result_rows = []

    for name, group in grouped_df:
        unique_chains = {}
        unique_ligands = set()
        mark = 0
        i=0
        for index, row in group.iterrows():
            ligand_list = []
            current_chain = row['chain']
            current_ligand = row['lig']
            ligand_list.append(current_ligand)
            if (current_chain in unique_chains.keys()) == False:
                if mark ==0:
                    unique_chains[current_chain] = ligand_list
                    mark=1
            else:
                unique_chains[current_chain].append(current_ligand)
        result_rows.append([name,list(unique_chains.keys())[0]])
        for lig in unique_chains[list(unique_chains.keys())[0]]:
            result_rows[-1].append(lig)
        i+=1
            # if current_ligand not in unique_ligands:
            #     # unique_chains.append(current_chain)
            #     unique_ligands.add(current_ligand)
            #     result_rows.append(row)
        # for chain_info in unique_chains:
        #     if chain_info

    # 创建新的 DataFrame
    result_df = pd.DataFrame(result_rows)

    # 保存结果到新的 CSV 文件
    result_df.to_csv(output_csv_path, index=False)

    print(f"已保存处理后的结果到 {output_csv_path}")

    # # 读取 CSV 文件
    # csv_file_path = '/home/g1/zyp/mymodel/dataset/holo4k_info_new_v1.csv'
    # df = pd.read_csv(csv_file_path)
    #
    # # 对相同 pdb 值的行进行分组，对 chain 进行收集
    # grouped_df = df.groupby('pdb')['chain'].agg(lambda x: ','.join(sorted(set(x)))).reset_index()
    #
    # # 如果不同 chain 对应的 lig 完全相同，则只保留一个 chain
    # unique_chains = []
    # for index, row in grouped_df.iterrows():
    #     pdb, chains = row['pdb'], row['chain'].split(',')
    #     unique_ligands = set(df[(df['pdb'] == pdb) & (df['chain'].isin(chains))]['lig'])
    #
    #     for chain in chains:
    #         ligands_for_chain = set(df[(df['pdb'] == pdb) & (df['chain'] == chain)]['lig'])
    #         if ligands_for_chain == unique_ligands:
    #             unique_chains.append(chain)
    #             break
    #
    # # 创建新的 DataFrame
    # result_df = pd.DataFrame({'pdb': grouped_df['pdb'], 'chain': unique_chains})
    #
    # # 保存结果到新的 CSV 文件
    # output_file_path = '/home/g1/zyp/mymodel/dataset/holo4k_info_new_v2.csv'
    # result_df.to_csv(output_file_path, index=False)
    #
    # print(f"已保存过滤后的结果到 {output_file_path}")
# 示例用法
# input_csv_path = '/home/g1/zyp/mymodel/dataset/holo4k_data/holo4k_info_remove_in_train_v1.csv'
# output_csv_path = '/home/g1/zyp/mymodel/dataset/holo4k_data/holo4k_info_remove_in_train_v2.csv'
# remove_chains(input_csv_path, output_csv_path)


def process_and_save_csv(input_csv_path, output_csv_path):
    # 读取 CSV 文件
    df = pd.read_csv(input_csv_path)

    # 处理 lig 列中的元素，只保留其4-7个字符
    df['lig'] = df['lig'].apply(lambda x: x[4:7] if isinstance(x, str) and len(x) >= 7 else x)

    # 保存结果到新的 CSV 文件
    df.to_csv(output_csv_path, index=False)

    print(f"已保存处理后的结果到 {output_csv_path}")

# 示例用法
input_csv_path = 'dataset/holo4k_data/holo4k_info_new.csv'
output_csv_path = 'dataset/holo4k_data/holo4k_info_new_v1.csv'

# process_and_save_csv(input_csv_path, output_csv_path)


import os
import pandas as pd
'''如果你想要比较一个文件夹下的文件名与一个 CSV 文件中的某一列，然后保存 CSV 文件中不重复的内容，你可以按照以下步骤进行操作：
读取 CSV 文件： 使用 Pandas 读取包含文件名的 CSV 文件。
获取文件夹下的文件名： 使用 Python 的 os 模块或其他文件操作模块，获取文件夹下的所有文件名。
比较并保存不重复的内容： 使用 Pandas 的 DataFrame 进行比较，并保存不重复的内容。'''

def compare_and_save_unique(csv_path, csv_column, folder_path, result_csv_path):
    # 读取 CSV 文件
    df_csv = pd.read_csv(csv_path)
    unique_pdb = []
    # 获取文件夹下的文件名
    folder_files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]

    for i,pdb_name in enumerate(df_csv[csv_column].values):
        mark = 0
        for file_name in folder_files:
            if pdb_name[0:4] == file_name[0:4]:
                mark = 1
        if mark == 0:
            unique_pdb.append(df_csv.values[i])
    # 比较并保存不重复的内容
    # unique_files = set(folder_files) - set(df_csv[csv_column])
    result_df = pd.DataFrame(unique_pdb)

    # 将结果保存为新的 CSV 文件
    result_df.to_csv(result_csv_path, index=False)

    print(f"Saved unique files to {result_csv_path}")

# 使用示例
csv_path = 'dataset/holo4k_data/holo4k_info_new_v1.csv'
csv_column = 'pdb'
folder_path = 'dataset/01-benchmark_pdbs'
result_csv_path = 'dataset/holo4k_data/holo4k_info_remove_in_train_v1.csv'

compare_and_save_unique(csv_path, csv_column, folder_path, result_csv_path)

import os
import pandas as pd
import shutil


def compare_csv_and_folder(csv_path, folder_path, column_name, new_folder_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 获取文件夹中的文件名
    folder_files = set(os.listdir(folder_path))

    # 找到文件夹中存在但不在 CSV 文件中的文件名
    files_to_copy = folder_files.intersection(set(df[column_name]))

    # 创建新文件夹（如果不存在）
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    # 复制文件到新文件夹
    for file_name in files_to_copy:
        src_file = os.path.join(folder_path, file_name)
        dest_file = os.path.join(new_folder_path, file_name)
        shutil.copy(src_file, dest_file)
        print(f"Copied {src_file} to {dest_file}")


# 使用示例
csv_path = 'dataset/holo4k_data/holo4k_info_remove_in_train_v2.csv'
folder_path = 'dataset/holo4k'
column_name = '0'
new_folder_path = 'dataset/holo4k_no_chongfu'

# compare_csv_and_folder(csv_path, folder_path, column_name, new_folder_path)


