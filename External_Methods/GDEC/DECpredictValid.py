import click
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
import uuid
import matplotlib.pyplot as plt
from util_brain import GeneDataSet, GeneNoGCNDataSet, GeneDotGCNDataSet
from ptdec.dec import DEC
from ptdec.model import train, predict
from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from ptdec.utils import cluster_accuracy
from sklearn.model_selection import train_test_split
@click.command()
@click.option(
    "--cuda", help="whether to use CUDA (default False).", type=bool, default=True
)
@click.option(
    "--batch-size", help="training batch size (default 256).", type=int, default=256
)
@click.option(
    "--pretrain-epochs",
    help="number of pretraining epochs (default 300).",
    type=int,
    default=200,
)
@click.option(
    "--finetune-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=300,
)
@click.option(
    "--trans-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=100,
)
@click.option(
    "--dec-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=50,
)
@click.option(
    "--testing-mode",
    help="whether to run in testing mode (default False).",
    type=bool,
    default=False,
)
def main(cuda, batch_size, pretrain_epochs, finetune_epochs, trans_epochs, dec_epochs, testing_mode):

    # callback function to call during training, uses writer from the scope
    # 找到真实标签
    filename='mouse10_100'
    root_path = f'../data/train/{filename}/'
    dataset_path = root_path + 'test_gene_express_label.csv'
    origin_data = pd.read_csv(dataset_path)
    origin_data = origin_data[['sample', 'label']]
    #找到其他模型预测的结果
    use_model = False
    pred_str = True
    if not use_model:
        # pred_data = pd.read_csv('../data/contrast/seurat/mouse_pan_merge(2).csv')
        # pred_data = pd.read_csv('../data/contrast/mona/pan_cluster_label.csv')
        # pred_data = pd.read_csv('../data/contrast/mona/brain_cluster_label.csv')
        pred_data = pd.read_csv('../data/contrast/seurat/mouse_brain_merge(1).csv')
        # pred_data = pd.read_csv('../data/contrast/scVI/brain_sc_cluster_label(1).csv')
        # pred_data = pd.read_csv('../data/contrast/scVI/pan_sc_cluster_label(1).csv')
        concat_data = pd.merge(origin_data,pred_data,left_on='sample',right_on='Cell',how='left').dropna().drop_duplicates()
        concat_data.drop(columns=['Cell'], inplace=True)
        concat_data.rename(columns={'Cell_type':'pred'},inplace=True)
        sure_label = pd.read_csv(root_path+'label_mapping_filtered.csv')
        label_mapping = sure_label._append({'label':len(sure_label),'CellType':'unknown','count':0},ignore_index=True)

        true = concat_data[['label']]
        true.rename(columns={'label':'CellType'}, inplace=True)
        true = pd.merge(true,label_mapping,on='CellType',how='left')
        true['label'].fillna(len(label_mapping)-1,inplace=True)
        true = np.asarray(true['label'],dtype='int64')

        if pred_str :
            predicted = concat_data[['pred']]
            predicted.rename(columns={'pred':'CellType'}, inplace=True)
            predicted = pd.merge(predicted, label_mapping, on='CellType', how='left')
            predicted['label'].fillna(len(label_mapping) - 1, inplace=True)
            predicted = np.asarray(predicted['label'], dtype='int64')
        else:
            predicted = concat_data[['pred']]
            predicted = np.asarray(predicted['pred'], dtype='int64')
    else:
        transfer_model = torch.load(f'../data/train/model/mouse10_100_gcn.pth')
        transfer_model.eval()
        data_pth = '../data/train/mouse10_100_gcn/valid_for_model.csv'
        ds_predict = GeneDotGCNDataSet(data_pth, testmode=True)
        predicted = pd.read_csv(data_pth)
        predicted = predicted[['sample']]
        predicted['pred'] = predict(
            ds_predict, transfer_model, 1024, silent=True, return_actual=False, cuda=cuda
        )
        pred_data = predicted.rename(columns={'sample':'Cell','pred':'Cell_type'})
        concat_data = pd.merge(origin_data, pred_data, left_on='sample', right_on='Cell',
                               how='left').dropna().drop_duplicates()
        true_mapping = pd.DataFrame(list(set(concat_data['label'].values)),columns=['CellType']).reset_index().rename(columns={'index':'label'})
        concat_data.drop(columns=['Cell'], inplace=True)
        concat_data.rename(columns={'Cell_type': 'pred'}, inplace=True)

        true = concat_data[['label']]
        true.rename(columns={'label': 'CellType'}, inplace=True)
        true = pd.merge(true, true_mapping, on='CellType', how='left')
        true = np.asarray(true['label'], dtype='int64')
        predicted = concat_data[['pred']]
        predicted = np.asarray(predicted['pred'], dtype='int64')



    reassignment, accuracy = cluster_accuracy(true, predicted)
    print("Final DEC accuracy: %s" % accuracy)

    predicted_reassigned = [
        reassignment[item] for item in predicted
    ]  # TODO numpify
    concat_data['clustering_res'] = predicted_reassigned
    confusion = confusion_matrix(true, predicted_reassigned)
    normalised_confusion = (
            confusion.astype("float") / confusion.sum(axis=1)[:, np.newaxis]
    )

    sns.heatmap(normalised_confusion)
    print()
    plt.show()
if __name__ == "__main__":
    main()
