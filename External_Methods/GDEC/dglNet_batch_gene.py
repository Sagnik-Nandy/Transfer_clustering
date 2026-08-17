import numpy as np
import pandas as pd
import torch as th
import dgl
from sklearn.decomposition import PCA


def make_graph(fea_dims=20,device=None,use_rnd_feas=False,kind='test'):
    """基因索引"""
    gene_node_DF = pd.read_csv(f'../data/net/gene_node_id.csv')
    gene_node_DF.rename(columns={'index':'gene_index','Symbl':'gene_symbl'}, inplace=True)
    gene_nums = gene_node_DF.shape[0]

    """蛋白质间作用边"""
    edge_gg_DF = pd.read_csv(f'../data/net/gene_interaction.csv')

    """构造蛋白质相互作用网络"""
    g_e_g_df = pd.merge(edge_gg_DF,gene_node_DF,how='inner', left_on='OfficalSymbl_x',right_on='gene_symbl')
    g_e_g_df.rename(columns={'gene_index':'u'},inplace=True)
    g_e_g_df = pd.merge(g_e_g_df, gene_node_DF,how='inner', left_on='OfficalSymbl_y', right_on='gene_symbl')
    g_e_g_df.rename(columns={'gene_index':'v'},inplace=True)
    g_e_g_df.dropna(inplace=True)

    p_e_p_lp_node = th.tensor(g_e_g_df['u'].values)
    p_e_p_rp_node = th.tensor(g_e_g_df['v'].values)

    rnd_feas = torch.randn(gene_nums, fea_dims)

    g = dgl.graph((p_e_p_lp_node,p_e_p_rp_node),num_nodes=gene_nums)
    bg = dgl.to_bidirected(g)
    bg.ndata['gene'] = rnd_feas
    print(g.ntypes)
    print(g.etypes)
    print(g.canonical_etypes)
    return bg

import dgl.function as fn
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch

class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_features, hidden_features)
        self.conv2 = dglnn.GraphConv(hidden_features, out_features)

    def forward(self, blocks, x):
        x = F.relu(self.conv1(blocks[0], x))
        x = F.relu(self.conv2(blocks[1], x))
        return x


class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']



class ScorePredictor(nn.Module):
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'))
            return edge_subgraph.edata['score']

class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.gcn = StochasticTwoLayerGCN(
            in_features, hidden_features, out_features)
        self.predictor = ScorePredictor()
    def forward(self, positive_graph, negative_graph, blocks, x):
        x = self.gcn(blocks, x)
        pos_score = self.predictor(positive_graph, x)
        neg_score = self.predictor(negative_graph, x)
        return pos_score, neg_score

def construct_negative_graph(graph, k):
    src, dst = graph.edges()
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())

def compute_loss(pos_score, neg_score):
    # 间隔损失
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()

def parse_node_embeddings(node_embeddings,kind):
    protein_np = node_embeddings.data.cpu().detach().numpy()
    protein_embedding_df = pd.DataFrame(protein_np).reset_index()
    protein_node_id_map = pd.read_csv(f'../data/net/gene_node_id.csv')
    protein_embedding_df = pd.merge(protein_node_id_map, protein_embedding_df, how='left', on='index')
    protein_embedding_df.to_csv(f'../data/net/gene_gcn_feas.csv', index=False)


if __name__ == '__main__':
    use_rnd_feas = True
    kind = 'Homo'
    fea_dims = 100
    hidden_fea_dims = 50
    out_fea_dims = 5
    k = 10
    epochs = 500
    min_loss, patient_nums, patient = 0, 50, 50
    batch_size = 10000

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    graph = make_graph(fea_dims, device, use_rnd_feas, kind)

    train_seeds = torch.arange(graph.num_edges()).to(device)

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler, negative_sampler=dgl.dataloading.negative_sampler.Uniform(5))
    use_uva = True
    dataloader = dgl.dataloading.DataLoader(
        graph,
        train_seeds ,
        sampler,
        device = device,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        use_uva=use_uva
        )

    agg_fn = 'mean'

    gene_feas = torch.tensor(graph.ndata['gene'], dtype=torch.float, device=device)


    import time
    start = time.time()
    model = Model(fea_dims, hidden_fea_dims, out_fea_dims)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(),
                           #lr=0.001
                           )
    for name, param in model.named_parameters():
        print(name, '-->', param.type(), '-->', param.dtype, '-->', param.shape)
    for epoch in range(epochs):
        loss_lst = []
        for idx, (input_nodes, positive_graph, negative_graph, blocks) in enumerate(dataloader):
            blocks = [b.to(torch.device(device)) for b in blocks]
            positive_graph = positive_graph.to(torch.device(device))
            negative_graph = negative_graph.to(torch.device(device))
            input_features = blocks[0].srcdata['gene']
            pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
            loss = compute_loss(pos_score, neg_score)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_lst.append(loss.item())
        loss_val = np.array(loss_lst).mean()
        if epoch == 0:
            min_loss = loss_val
        elif loss_val < min_loss:
            min_loss = loss_val
            patient = patient_nums
        else:
            patient -= 1
        print(f'epoch:{epoch}:loss:{round(loss.item(),4)}')
        if patient == 0: break
    end = time.time()
    print(f'total time {end - start}s')

    #保存模型
    path = f'../model/{kind}_dgl_model_homo.pt'
    torch.save(model.state_dict(), path)
    graph = graph.to(device)
    node_embeddings = model.gcn([graph, graph], gene_feas)
    parse_node_embeddings(node_embeddings,kind)
    print()
