import pandas as pd


def make_id_for_net():
    interaction = pd.read_csv(f'../data/net/HumanNet-GSP.tsv', sep='\t',names=['gene1','gene2'])
    gene_id = set(interaction['gene1'].tolist() + interaction['gene2'].tolist())
    gene_id = sorted(list(gene_id))
    gene_id = pd.DataFrame(gene_id,columns=['EntrezId'])
    gene_id.reset_index(inplace=True)
    gene_id.rename(columns={'index':'node_id'}, inplace=True)
    gene_id['EntrezId'].to_csv(r'../data/net/EntrezId.csv', index=False)

    # 在DAVID网站获得对应的id映射
    id_mapping = pd.read_csv('../data/gene_id_mapping.txt', sep='\t',header=0)
    gene_id = pd.merge(gene_id,id_mapping,left_on='EntrezId',right_on='From',how='inner')
    gene_id.rename(columns={'To':'OfficalSymbl'},inplace=True)
    gene_mapping = gene_id[['EntrezId','OfficalSymbl']]
    interaction = pd.merge(interaction, gene_mapping,left_on='gene1',right_on='EntrezId',how='inner')
    interaction = pd.merge(interaction, gene_mapping,left_on='gene2',right_on='EntrezId',how='inner')
    interaction[['OfficalSymbl_x','OfficalSymbl_y']].to_csv(r'../data/net/gene_interaction.csv',index=False)
    gene_node_id = list(set(interaction['OfficalSymbl_x'].tolist() + interaction['OfficalSymbl_y'].tolist()))
    gene_node_id = pd.DataFrame(gene_node_id,columns=['Symbl'])
    gene_node_id = gene_node_id.sort_values(by=['Symbl'], ignore_index=True).reset_index()
    gene_node_id.to_csv(r'../data/net/gene_node_id.csv', index=False)
    print()
if __name__ == '__main__':
    make_id_for_net()