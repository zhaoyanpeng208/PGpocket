import Bio
from Bio.PDB import *
from Bio.PDB.PDBParser import PDBParser
from masif_opts import masif_opts
import os
import pandas as pd
import numpy as np
from input_output.extractPDB import extractPDB
from tqdm import trange,tqdm
from triangulation.xyzrn import output_pdb_as_xyzrn
# masif_opts["raw_pdb_dir"]


# 将配体的结构pdb文件单独提取出来
def read_all_ligand(pdb, chain_ID, ligand_list):
    # pdb = '/home/g1/zyp/IDBPG/data_preparation/data_preparation/00-raw_pdbs/' + '1a2n.pdb'
    pdb_id = masif_opts["raw_pdb_dir"] + pdb + '.pdb'
    parser = PDBParser()
    structure = parser.get_structure(pdb, pdb_id)
    model = structure[0]
    chain = model[chain_ID.rstrip("\n")]
    for residue in chain.get_list():
        residue_id = residue.get_id()
        hetfield = residue_id[0]
        if hetfield != '' and hetfield[2:] in ligand_list:
            if not os.path.exists(masif_opts["raw_pdb_dir"] + 'ligand/' + pdb + '_' + str(hetfield[2:]) + '.pdb'):
                print(pdb,hetfield)
                io = PDBIO()
                io.set_structure(residue)
                io.save(masif_opts["raw_pdb_dir"] + 'ligand/' + pdb +'_' + str(hetfield[2:]) + '.pdb')
            # with open('1a2n'+'_'+hetfield+'.pdb', 'w') as ligandfile:
            #     ligandfile.write(str(residue))
            #     ligandfile.close()


# 通过计算距离得到正样本
def cal_distance(pdb, chain_ID, ligand_list):
    ligand = []
    pdb_id = masif_opts["raw_pdb_dir"] + pdb + '.pdb'
    parser = PDBParser()
    structure = parser.get_structure(pdb, pdb_id)
    model = structure[0]
    chain = model[chain_ID.rstrip("\n")]
    for residue in chain.get_list():
        residue_id = residue.get_id()
        hetfield = residue_id[0]
        if hetfield != '' and hetfield[2:] in ligand_list:
            ligand.append(residue)
    for l in ligand:
        pos_atom = []
        for atom_l in l:
            for residue in chain:
                if is_aa(residue):
                    dis = atom_l - residue['CA']
                    if dis <= 7:
                        for atom in residue:
                            dis = atom_l - atom
                            if dis <= 4 and atom not in pos_atom:
                                pos_atom.append(atom)


# 上面两个函数的主函数
def main():
    # rawpdb = masif_opts["raw_pdb_dir"]+'1a2n.pdb'
    pdblists = pd.read_csv(r'/home/g1/zyp/IDBPG/data_preparation/data_preparation/holo4k_info_remove_in_train_v2.csv'
                           , header=None)
    pdblists.reset_index()
    # print(pdblists.shape)
    for i in trange(1, pdblists.shape[0]):
        # Save the chains as separate files.
        pdblist = pdblists.loc[i].values[0:-1]
        pdblist = [a_ for a_ in pdblist if a_ == a_]
        temp = pdblist[3].split('_')
        pdb_id = temp[0]
        chain_ids1 = temp[1]
        j = 4
        ligand_list = []
        while j < len(pdblist):
            ligand_list.append(pdblist[j])
            j += 1

        cal_distance(pdb_id,chain_ids1,ligand_list)
        read_all_ligand(pdb_id, chain_ids1,ligand_list) #提取配体PDB
        pdb_filename = masif_opts['raw_pdb_dir'] + pdb_id + ".pdb"
        out_filename1 = masif_opts['pdb_chain_dir']+"/"+pdb_id+"_"+chain_ids1
        extractPDB(pdb_filename, out_filename1+".pdb", chain_ids1)# 提取需要的pdb链


# 转换pdb为xyzrn
def pdb2xyzrn():
    file_base = masif_opts['raw_pdb_dir'] + "msms/"
    pdbroot = masif_opts['raw_pdb_dir']+'positive/'
    list = os.listdir(pdbroot)
    for dir in tqdm(list):
        out_xyzrn = dir.split('.')[0] + ".xyzrn"
        pdbfilename = pdbroot + dir
        xyzrnfilename = file_base + out_xyzrn
        output_pdb_as_xyzrn(pdbfilename, xyzrnfilename)


# pdb2xyzrn()
main()
# extractPDB('data_preparation/00-raw_pdbs/1bx9.pdb', 'data_preparation/01-benchmark_pdbs/1bx9.pdb','A')
# posroot = masif_opts['raw_pdb_dir'] + 'positive/'
# lists = os.listdir(posroot)
# bug_list = []
# for list in lists:
#     posfile = open(posroot + list)
#     meshdata = (posfile.read().rstrip()).split("\n")
#     if(len(meshdata) < 20):
#         bug_list.append(list.split('.')[0])
# print(bug_list)
# # bug_list.append('2aaz')
# df=pd.DataFrame(bug_list)
# df.to_csv('bug_list', index=None)

def find_repeat_data(name_list):
    """
    查找列表中重复的数据
    :param name_list:
    :return: 一个重复数据的列表，列表中字典的key 是重复的数据，value 是重复的次数
    """
    repeat_list = []
    for i in set(name_list):
        ret=name_list.count(i) # 查找该数据在原列表中的个数
        if ret > 1:
            item=dict()
            item[i] = ret
            repeat_list.append(item)
    return repeat_list


# root = '/home/g1/zyp/IDBPG/data_preparation/data_preparation/01-benchmark_surfaces'
# list = os.listdir(root)
#
# print(find_repeat_data(list))
