import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS, KMeans, AgglomerativeClustering
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

"""
根据预测结果进行聚类并计算DCC
"""
# 预测结果文件，由predict得到
file_path = './dataset/pred_dic_08-12_19-13-52.npy'
# 保存蛋白质链的文件夹，用于可视化
input_pdb_file = './dataset/test_pdb/processed/'
# 可视化后保存的文件
output_pdb_file = './dataset/test_pdb/draw_full'
# 真实位点的文件，前者是配体中心文件，后者是位点中心文件
dis_pdb_file = './dataset/test_pdb/lig_coord_dic.npy'
# dis_pdb_file = './dataset/true_site_test_dic.npy'
def get_first_elements(matrix):
    return [row[0] for row in matrix]
def get_second_elements(matrix):
    return [row[1] for row in matrix]

def cal_mean_points(points):
    x = [row[0] for row in points]
    y = [row[1] for row in points]
    z = [row[2] for row in points]
    mean_x = (max(x)+min(x))/2
    mean_y = (max(y)+min(y))/2
    mean_z = (max(z) + min(z)) / 2
    return np.array([mean_x,mean_y,mean_z])


def cal_dcc(pred_dic, dis_dic):
    distance_list = []
    count_clu = 0
    for pdb in pred_dic:
        try:
            distance = 100
            # if pdb == '4ccz_A':
                # print(pdb)
            clu_res = {}
            clu_p ={}
            clu_mean_info=[]
            true_site_list = dis_dic[pdb]
            site_num = len(true_site_list)
            pred_sites_info = pred_dic[pdb][1]
            pred_sites = get_first_elements(pred_sites_info)
            p_list = get_second_elements(pred_sites_info)
            optics_model = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
            optics_model.fit(pred_sites)
            labels = optics_model.labels_
            # for label, feature in zip(labels, pred_sites):
            #     if label not in clu_res:
            #         clu_res[label] = [feature]
            #     else:
            #         clu_res[label].append(feature)

            # agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1, linkage='ward')
            # y_agg = agg_clustering.fit_predict(pred_sites)
            # labels = y_agg.labels_
            # n = int(len(pred_sites)/100)+2
            # kmeans = KMeans(n_clusters=n, random_state=42)
            # y_kmeans = kmeans.fit(pred_sites)
            # labels = y_kmeans.labels_
            for label, feature, prob in zip(labels, pred_sites,p_list):
                if label not in clu_res:
                    clu_res[label] = [feature]
                    clu_p[label] = [prob]
                else:
                    clu_res[label].append(feature)
                    clu_p[label].append(prob)
            clu_num = len(clu_res)
            count_clu+=clu_num
            for clu in clu_res:
                if clu != -1:
                    mean_point = np.mean(clu_res[clu],axis=0)
                    mean_p = np.mean(clu_p[clu],axis=0)
                    max_p = np.max(clu_p[clu],axis=0)
                    len_clu = len(clu_res[clu])
                    clu_mean_info.append([mean_point,len_clu])
                    # mean_point = cal_mean_points(clu_res[clu])
            sorted_list = sorted(clu_mean_info, key=lambda x: x[1],reverse=True)[:site_num]
            for true_site in true_site_list:
                center_site = np.mean(true_site, axis=0)
                for pred_site in sorted_list:
                    temp = np.linalg.norm(pred_site[0] - center_site)
                    if temp < distance:
                        distance = temp
                distance_list.append(distance)
                print(pdb, distance)
        except:
            print(pdb)
    print(count_clu)

# print(distance_list)

    # 绘制原始数据和聚类结果
    # colors = [plt.cm.nipy_spectral(each) for each in np.linspace(0, 1, len(set(labels)))]
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(len(pred_sites)):
    #     ax.scatter(pred_sites[i][0], pred_sites[i][1], pred_sites[i][2],color=colors[labels[i]], s=10)
    #
    # plt.title('OPTICS Clustering')
    # plt.show()
dis_dic = np.load(dis_pdb_file, allow_pickle=True).item()
# for i in range(1,18):
pred_dic = np.load('./dataset/07-12_16-56-38/pred_dic_'+str(15)+'07-12_16-56-38.npy', allow_pickle=True).item()
cal_dcc(pred_dic,dis_dic)


