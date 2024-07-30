import numpy as np
# import pandas as pd
from Bio import PDB
from Bio.PDB import Selection
from Bio.PDB.PDBParser import PDBParser
import os
from matplotlib import pyplot as plt
import pandas as pd

def vis(read_pdb_path, write_pdb_path, points):
    f = open(read_pdb_path,"r")   #设置文件对象
    data = f.readlines()  #直接将文件中按行读到list里
    res_no = 0
    with open(write_pdb_path, 'w') as w_f:
        for line in data:
            if len(line) == 81 and (line[0:4]=='ATOM'or line[0:6]=='HETATM'):
                atom_no = line[6:11].strip(' ')
                atom_no = int(atom_no)
                atom = (line[13:16].strip(' '))
                res = (line[17:20])
                chain_ID = line[21:26]
                chain = chain_ID[0]
                res_no = chain_ID[1:].strip(' ')
                res_no = int(res_no)
                chain_ID = (chain, res_no)
                w_f.write(line)
        for point in points:
            coord = point[0]
            p = point[1]
            # coord = points
            # p = 0
            atom_no+=1
            res_no+=1
            chain = chain
            # new_pos = [float(line1[30:37]), float(line1[38:45]), float(line1[46:53])]
            # for i in range(len(attent_raw)):
            #     if attent_raw[i][1] == res_no:
            w_f.write('HETATM'+str(atom_no).rjust(5,' ')+ ' MG    MG '+chain+
                      str(res_no).rjust(4,' ')+'     '+ str(format(coord[0], '.3f')).rjust(7,' ')+
                      str(format(coord[1], '.3f')).rjust(8,' ')+str(format(coord[2], '.3f')).rjust(8,' ')
                      + '  1.00  '+ str(format(p, '.2f')) +'          '+str("MG")+'\n')

def draw_pred_pdb(file_path,input_pdb_file,output_pdb_file):
    pred_dic = np.load(file_path, allow_pickle=True).item()
    for pdb in pred_dic:
        # pdb_id = pdb.split('_')[0]
        # chain = pdb.split('_')[1]
        pdb_id = pdb[0:4]
        chain = pdb[5]
        points = pred_dic[pdb][1]
        # try:
        # vis(input_pdb_file+"pdb%s.ent" % (pdb_id),output_pdb_file+"pdb%s.ent" % (pdb_id), points)
        vis(input_pdb_file+pdb_id+'.pdb', output_pdb_file + "%s.pdb" % (pdb_id), points)
        # except:
        #     print(pdb)

file_path = './dataset/07-12_16-56-38/holo4k_15.npy'
input_pdb_file = './dataset/holo4k_no_chongfu/'
output_pdb_file = './dataset/holo4k_data/vis_pdb/'
draw_pred_pdb(file_path, input_pdb_file,output_pdb_file)


def get_coordinates_from_pdb(pdb_file, molecule_name):
    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure('structure', pdb_file)
    except PDB.PDBExceptions.PDBConstructionWarning as e:
        print(f"Error reading file {pdb_file}: {e}")
        return []

    coordinates_list = []

    for model in structure:
        for chain in model:
            for residue in chain:
                coordinates = []
                if residue.id[0][0]=='H' and residue.resname == molecule_name:
                    for atom in residue:
                        coordinates.append(atom.coord)
                if coordinates!=[]:
                    coordinates_list.append(coordinates)

    return coordinates_list

def process_pdb_files(folder_path, molecule_name):
    """
    获取配体分子的原子坐标集合
    :param folder_path: 蛋白质文件
    :param molecule_name: 配体名称文件
    :return:
    """
    lig_coord_dic={}
    # 获取文件夹中后缀为.pdb/.ent的文件
    # pdb_files = [f for f in os.listdir(folder_path) if f.endswith(".ent")]
    pdb_files = [f for f in os.listdir(folder_path) if f.endswith(".pdb")]
    for pdb_file in pdb_files:
        total_coord_list = []
        pdb_file_path = os.path.join(folder_path, pdb_file)
        lig_list = molecule_name[pdb_file]
        lig_list = list(set(lig_list))
        for lig in lig_list:
            coordinates = get_coordinates_from_pdb(pdb_file_path, lig)
            if coordinates:
                print(f"Coordinates from {pdb_file}: {coordinates}")
            else:
                print(f"No coordinates found in {pdb_file} for molecule {molecule_name}")
            for c in coordinates:
                total_coord_list.append(c)
        lig_coord_dic[pdb_file] = total_coord_list

    np.save('./dataset/test_pdb/coach_lig_coord_dic.npy',lig_coord_dic)

# # 示例用法
# folder_path = './dataset/coach420/'
# file_path = './dataset/coach420_data/pdb_lig_coach420.csv'
# df = pd.read_csv(file_path)
#
# # 选择某两列
# selected_columns = df
# lig_dic = {}
# for colums in selected_columns.values:
#     temp_lig_list = []
#     pdb_name = colums[1]
#     for lig in colums[2:]:
#         if str(lig) != 'nan':
#             temp_lig_list.append(lig)
#     lig_dic[pdb_name] = temp_lig_list
#
# process_pdb_files(folder_path, lig_dic)


def vis_true_site(read_pdb_path, write_pdb_path, true_lig,true_site,pred_sites):
    #根据vis改的
    f = open(read_pdb_path,"r")   #设置文件对象
    data = f.readlines()  #直接将文件中按行读到list里
    res_no = 0
    with open(write_pdb_path, 'w') as w_f:
        for line in data:
            if len(line) == 81:
                atom_no = line[6:11].strip(' ')
                atom_no = int(atom_no)
                atom = (line[13:16].strip(' '))
                res = (line[17:20])
                chain_ID = line[21:26]
                chain = chain_ID[0]
                res_no = chain_ID[1:].strip(' ')
                res_no = int(res_no)
                chain_ID = (chain, res_no)
                w_f.write(line)
        # for point in points:
        lig_coord = true_lig
        p = 0
        atom_no+=1
        res_no+=1
        chain = chain
        w_f.write('HETATM'+str(atom_no).rjust(5,' ')+ ' AG    AG '+chain+
                  str(res_no).rjust(4,' ')+'     '+ str(format(lig_coord[0], '.3f')).rjust(7,' ')+
                  str(format(lig_coord[1], '.3f')).rjust(8,' ')+str(format(lig_coord[2], '.3f')).rjust(8,' ')
                  + '  1.00  '+ str(format(p, '.2f')) +'          '+str("AG")+'\n')
        true_site_coord = true_site
        p = 0
        atom_no += 1
        res_no += 1
        chain = chain
        w_f.write('HETATM' + str(atom_no).rjust(5, ' ') + ' CU    CU ' + chain +
                  str(res_no).rjust(4, ' ') + '     ' + str(format(true_site_coord[0], '.3f')).rjust(7, ' ') +
                  str(format(true_site_coord[1], '.3f')).rjust(8, ' ') + str(format(true_site_coord[2], '.3f')).rjust(8, ' ')
                  + '  1.00  ' + str(format(p, '.2f')) + '          ' + str("CU") + '\n')
        for point in pred_sites:
            coord = point[0]
            p = point[1]
            # coord = points
            # p = 0
            atom_no += 1
            res_no += 1
            chain = chain
            w_f.write('HETATM' + str(atom_no).rjust(5, ' ') + ' MG    MG ' + chain +
                      str(res_no).rjust(4, ' ') + '     ' + str(format(coord[0], '.3f')).rjust(7, ' ') +
                      str(format(coord[1], '.3f')).rjust(8, ' ') + str(format(coord[2], '.3f')).rjust(8, ' ')
                      + '  1.00  ' + str(format(p, '.2f')) + '          ' + str("MG") + '\n')

def true_site_dis_cal(lig_path,site_path,pred_site_path):
    #可视化，将三个输入文件中的坐标画在一个pdb中
    lig_dic = np.load(lig_path,allow_pickle=True).item()
    site_dic = np.load(site_path,allow_pickle=True).item()
    pred_dic= np.load(pred_site_path,allow_pickle=True).item()
    input_pdb_file = './dataset/coach420/'
    output_pdb_file = './dataset/coach420_data/vis_pdb/'
    dis_dic = {}
    # print(lig_dic)
    for lig in lig_dic:
        dis = np.linalg.norm(np.mean(lig_dic[lig][0],axis=0) - site_dic[lig][0])
        # print(lig,dis)
        dis_dic[lig] = [np.mean(lig_dic[lig][0],axis=0),site_dic[lig][0],dis]
        try:
            vis_true_site(input_pdb_file + "pdb%s.pdb" % (lig), output_pdb_file + "pdb%s.pdb" % (lig),
                          np.mean(lig_dic[lig][0],axis=0),site_dic[lig][0],pred_dic[lig][1])
            # vis(output_pdb_file + "pdb%s.ent" % (lig), output_pdb_file + "pdb%s.ent" % (lig), site_dic[lig][1])
        except:
            print(lig)
    # np.save('./dataset/test_pdb/true_site_dis_cal_dic.npy', dis_dic)

# # 保存配体中心坐标的文件
# lig_path='./dataset/test_pdb/coach_lig_coord_dic.npy'
# # 保存结合位点中心坐标的文件
# site_path='./dataset/test_pdb/coach_lig_coord_dic.npy'
# # 保存预测口袋坐标的文件
# pred_site_path = './dataset/07-12_16-56-38/coach420.npy'
# # 可视化，将上述三个文件中的坐标画在一个pdb中
# true_site_dis_cal(lig_path,site_path,pred_site_path)



