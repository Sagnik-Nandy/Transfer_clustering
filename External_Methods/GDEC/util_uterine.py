# 这是一个示例 Python 脚本。
from collections import Counter

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils import data
def concat_express_and_label_and_gcn_data(file_name = 'ssr1'):
    source_gene_express = pd.read_csv('../data/uterine/Myometrium/new2_M_merge.csv', index_col=0)
    target_gene_express = pd.read_csv('../data/uterine/Myometrium/My_merge.csv', index_col=0)
    gene_gcn = pd.read_csv('../data/net/gene_gcn_feas.csv')
    src_label = pd.read_csv('../data/uterine/Myometrium/new2_meta.csv')
    tar_label = pd.read_csv('../data/uterine/Myometrium/My_meta.csv')
    source_gene_express.reset_index(inplace=True)
    source_gene_express['index'] = source_gene_express['index'].apply(lambda x: str(x).lower())
    target_gene_express.reset_index(inplace=True)
    target_gene_express['index'] = target_gene_express['index'].apply(lambda x: str(x).lower())
    gene_gcn.drop(columns=['index'], inplace=True)
    gene_gcn['Symbl'] = gene_gcn['Symbl'].apply(lambda x: str(x).lower())

    source_gene_set = set(source_gene_express['index'].tolist())
    target_gene_set = set(target_gene_express['index'].tolist())
    gcn_gene_set = set(gene_gcn['Symbl'].tolist())
    common_gene_set = list(source_gene_set & target_gene_set & gcn_gene_set)

    source_gene_express = source_gene_express[source_gene_express['index'].isin(common_gene_set)]
    source_gene_express = source_gene_express.sort_values(by=['index'])
    target_gene_express = target_gene_express[target_gene_express['index'].isin(common_gene_set)]
    target_gene_express = target_gene_express.sort_values(by=['index'])
    gene_gcn = gene_gcn[gene_gcn['Symbl'].isin(common_gene_set)]

    gene_gcn_scala = gene_gcn.set_index('Symbl')
    gene_gcn_scala = gene_gcn_scala.T
    gene_gcn_scala = (gene_gcn_scala - gene_gcn_scala.min()) / (gene_gcn_scala.max() - gene_gcn_scala.min())
    gene_gcn_scala = gene_gcn_scala.T
    gene_gcn_scala.reset_index(inplace=True)
    gene_gcn_scala.rename(columns={'index':'Symbl'}, inplace=True)
    gene_gcn_scala.to_csv(fr'../data/train/{file_name}/gcn_scala_feas.csv', index=False)

    source_gene_express.set_index('index', inplace=True)
    target_gene_express.set_index('index', inplace=True)
    source_gene_express = (source_gene_express - source_gene_express.min()) / (
            source_gene_express.max() - source_gene_express.min())
    target_gene_express = (target_gene_express - target_gene_express.min()) / (
                target_gene_express.max() - target_gene_express.min())

    # pd.DataFrame(source_gene_express['index'].tolist(),columns=['gene']).to_csv(fr'../data/train/{file_name}/gene_for_train.csv', index=False)


    source_gene_express = source_gene_express.T
    source_gene_express.reset_index(inplace=True)

    target_gene_express = target_gene_express.T
    target_gene_express.reset_index(inplace=True)

    source_gene_express_label = pd.merge(source_gene_express,src_label,left_on='index',right_on='Cell',how='inner')
    source_gene_express_label.rename(columns={'Cell': 'sample'}, inplace=True)
    target_gene_express_label = pd.merge(target_gene_express, tar_label, left_on='index', right_on='Cell', how='inner')
    target_gene_express_label.rename(columns={'Cell': 'sample'}, inplace=True)

    # 剔除数量过少的
    label_count = source_gene_express_label[['Cell_type','index']].groupby(by=['Cell_type']).count()
    target_label_count = target_gene_express_label[['Cell_type', 'index']].groupby(by=['Cell_type']).count()
    label_count = label_count[label_count['index'] > 0]
    label_count = label_count.reset_index()
    label_count.rename(columns={'Cell_type':'CellType','index':'count'}, inplace=True)
    label_count.reset_index(inplace=True)
    label_count.rename(columns={'index':'label'}, inplace=True)
    label_count.to_csv(fr'../data/train/{file_name}/label_mapping_filtered.csv',index=False)

    source_gene_express_label = pd.merge(source_gene_express_label,label_count,left_on='Cell_type',right_on='CellType')
    source_gene_express_label.drop(columns=['Cell_type','CellType','index','count'], inplace=True)
    source_gene_express_label.to_csv(fr'../data/train/{file_name}/source_gene_express_label.csv', index=False)

    valid_gene_express_label = pd.merge(target_gene_express_label, label_count, left_on='Cell_type',
                                         right_on='CellType')
    valid_gene_express_label.drop(columns=['Cell_type', 'CellType', 'index', 'count'], inplace=True)
    valid_gene_express_label.to_csv(fr'../data/train/{file_name}/valid_gene_express_label.csv', index=False)

    target_gene_express_label.drop(columns=['index'], inplace=True)
    target_gene_express_label.rename(columns={'Cell_type':'label'}, inplace=True)
    target_gene_express_label.to_csv(fr'../data/train/{file_name}/test_gene_express_label.csv', index=False)

def make_dot_features(gene_express_path = '../data/train/dataset_human_mus/source_gene_express_label.csv',
                      gcn_path = '../data/train/dataset_human_mus/gcn_scala_feas.csv',
                      target_path = '../data/train/dataset_human_mus/source_for_model.csv',
                      zipgcn=True):
    gene_express = pd.read_csv(gene_express_path)
    label_df = gene_express[['label','sample']]
    gene_express.drop(columns=['label','sample'], inplace=True)

    gene_express = gene_express.T
    gene_express.reset_index(inplace=True)
    gcn = pd.read_csv(gcn_path)
    gcn['Symbl'] = gcn['Symbl'].apply(lambda x: str(x).lower())
    gene_exp_gcn = pd.merge(gene_express, gcn, left_on='index', right_on='Symbl', how='left')
    gene_exp_gcn.fillna(0,inplace=True)
    gene_exp_gcn_tmp = gene_exp_gcn.drop(columns=['Symbl','index'])
    exp_tensor = torch.tensor(np.asarray(gene_exp_gcn_tmp.iloc[:,:-20]))
    gcn_tensor = torch.tensor(np.asarray(gene_exp_gcn_tmp.iloc[:,-20:]))
    dot_tensor = torch.mm(exp_tensor.T,gcn_tensor)
    gene_express.set_index('index',inplace=True)
    gene_express = gene_express.T
    dot_df = pd.DataFrame(np.asarray(dot_tensor))
    if zipgcn:
        dot_df = dot_df.sum(axis=1)
    dot_df = (dot_df - dot_df.min()) / (dot_df.max()  - dot_df.min())
    gene_express_dot = pd.concat([gene_express,dot_df,label_df],axis=1)
    gene_express_dot.to_csv(target_path, index=False)


class GeneDataSet(data.Dataset):

    def __init__(self, gene_express_path, gcn_path,target_gene_path):
        gene_express = pd.read_csv(gene_express_path)
        self.gcn_feas = pd.read_csv(gcn_path)
        self.genes = pd.read_csv(target_gene_path)
        self.label = gene_express['label']
        self.feas = gene_express.iloc[:,:-1]

    def make_matrix(self,gene_express,gcn,index):

        gcn['Symbl'] = gcn['Symbl'].apply(lambda x:str(x).lower())
        gene_exp = pd.DataFrame(gene_express.iloc[index,:])
        gene_exp.rename(columns={index:'express'}, inplace=True)
        gene_exp.reset_index(inplace=True)
        gene_exp_gcn = pd.merge(gene_exp,gcn,left_on='index',right_on='Symbl',how='left')
        gene_exp_gcn.drop(columns=['Symbl'],inplace=True)
        gene_exp_tensor = torch.tensor(gene_exp_gcn['express'])
        gene_gcn_feas = np.asarray(gene_exp_gcn.iloc[:,2:])
        gene_gcn_tensor = torch.tensor(gene_gcn_feas)
        gene_exp_tensor = torch.unsqueeze(gene_exp_tensor,dim=1)
        gene_exp_gcn_tensor = gene_gcn_tensor + gene_exp_tensor
        if not ((gene_exp_gcn_tensor.shape[0] == 1109) & (gene_exp_gcn_tensor.shape[1] == 20)):
            print('error!!!!!!!!')
        return gene_exp_gcn_tensor

    def __getitem__(self, index):
        feature = self.make_matrix(self.feas, self.gcn_feas,index)
        feature = torch.reshape(feature,[-1])
        label = np.asarray(self.label)
        label = torch.tensor(label[index])
        feature = feature.to(torch.float32)
        label = label.to(torch.float32)
        return feature, label


    def __len__(self):
        return len(self.label)




class GeneNoGCNDataSet(data.Dataset):

    def __init__(self, gene_express_path, gcn_path,target_gene_path):
        gene_express = pd.read_csv(gene_express_path)
        self.gcn_feas = pd.read_csv(gcn_path)
        self.genes = pd.read_csv(target_gene_path)
        self.label = gene_express['label']
        self.feas = gene_express.iloc[:,:-1]

    def __getitem__(self, index):
        label = np.asarray(self.label)
        label = torch.tensor(label[index])
        feature = np.asarray(self.feas)
        feature = torch.tensor(feature[index,:])
        feature = feature.to(torch.float32)
        label = label.to(torch.float32)
        return feature, label


    def __len__(self):
        return len(self.label)

class GeneDotGCNDataSet(data.Dataset):

    def __init__(self, gene_express_path, testmode=False):
        gene_express = pd.read_csv(gene_express_path)
        self.label = gene_express['label']
        self.feas = gene_express.iloc[:,:-1]
        self.test = testmode
    def __getitem__(self, index):
        feature = np.asarray(self.feas.iloc[index,:])
        feature = torch.tensor(np.asarray(feature))
        feature = feature.to(torch.float32)

        if self.test:

            return feature
        else:
            label = np.asarray(self.label)
            label = torch.tensor(label[index])
            label = label.to(torch.float32)
            return feature, label


    def __len__(self):
        return len(self.label)
# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    pass

    filename = 'new2_M'
    concat_express_and_label_and_gcn_data(filename)


    plus = '_gcn'
    zipgcn = True
    gene_express_path = f'../data/train/{filename}/source_gene_express_label.csv'
    gcn_path = f'../data/train/{filename}/gcn_scala_feas.csv'
    target_path = f'../data/train/{filename}{plus}/source_for_model.csv'
    make_dot_features(gene_express_path,gcn_path,target_path,zipgcn)
    gene_express_path = f'../data/train/{filename}/valid_gene_express_label.csv'
    target_path = f'../data/train/{filename}{plus}/valid_for_model.csv'
    make_dot_features(gene_express_path, gcn_path, target_path, zipgcn)
    gene_express_path = f'../data/train/{filename}/test_gene_express_label.csv'
    target_path = f'../data/train/{filename}{plus}/test_for_model.csv'
    make_dot_features(gene_express_path, gcn_path, target_path, zipgcn)

