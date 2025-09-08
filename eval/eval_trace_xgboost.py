import os
import pandas as pd
import torch
import numpy as np
import joblib
import json
from tqdm import tqdm
from gensim.models import Word2Vec
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric import utils

from model.GAT import GAT
from model.TGN import TemporalGraphNetwork
from model.MultiHeadSelfAttention import MultiHeadSelfAttentionModel
from model.PositionalEncoder import PositionalEncoder
from utils.graphutils_trace import add_attributes, prepare_graph
from utils.helper import helper

# 加载 Word2Vec 模型和编码器
w2vmodel = Word2Vec.load("../train/word2vec_trace.model")
encoder = PositionalEncoder(30)

def infer_word2vec_embedding(document):
    word_embeddings = [w2vmodel.wv[word] for word in document if word in w2vmodel.wv]
    if not word_embeddings:
        return np.zeros(30)
    output_embedding = np.array(word_embeddings)
    output_embedding = torch.tensor(output_embedding, dtype=torch.float)
    output_embedding = encoder.embed(output_embedding)
    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")


    with open("../data/trace/trace_test.txt") as f:
        data = f.read().split('\n')
    data = [line.split('\t') for line in data if line]
    df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
    df = df.dropna()
    df.sort_values(by='timestamp', ascending=True, inplace=True)

    df = add_attributes(df, "../data/trace/ta1-trace-e3-official-1.json.4")
    with open("../data/trace/trace.json", "r") as json_file:
        GT_mal = set(json.load(json_file))

    phrases, labels, edges, mapp, node_types, edges_attr = prepare_graph(df)
    all_ids = set(df['actorID']).union(set(df['objectID']))

    graph = Data(
        x=torch.tensor(np.array(node_types), dtype=torch.float).to(device),
        y=torch.tensor(labels, dtype=torch.long).to(device),
        edge_index=torch.tensor(edges, dtype=torch.long).to(device),
        edge_time=torch.tensor(edges_attr, dtype=torch.float).to(device)
    )
    graph.n_id = torch.arange(graph.num_nodes).to(device)
    flag = torch.ones(graph.num_nodes, dtype=torch.bool).to(device)


    num_classes = 12
    gat_model = GAT(in_channel=12, out_channel=30).to(device)
    tgn_model = TemporalGraphNetwork(in_channels=60, out_channels=32).to(device)
    att_model = MultiHeadSelfAttentionModel(in_channels=32, out_channels=12, num_classes=num_classes, num_heads=4).to(device)
    xgb_model = joblib.load('../trained_weights/trace_xgb/xgb_model.pkl')

    gat_model.load_state_dict(torch.load(f'../trained_weights/trace_xgb/gat_final.pth', map_location=device))
    tgn_model.load_state_dict(torch.load(f'../trained_weights/trace_xgb/tgn_final.pth', map_location=device))
    att_model.load_state_dict(torch.load(f'../trained_weights/trace_xgb/selfattention_final.pth', map_location=device))

    for epoch in range(30):
        print(f"Epoch: {epoch}")

        gat_model.eval()
        tgn_model.eval()
        att_model.eval()
        loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=26000)
        for subg in tqdm(loader, desc="Evaluating", leave=False):
            subg = subg.to(device)
            current_phrases = [phrases[idx] for idx in subg.n_id.tolist()]
            word2vec_embeddings = [infer_word2vec_embedding(p) for p in current_phrases]
            word2vec_tensor = torch.tensor(np.array(word2vec_embeddings), dtype=torch.float).to(device)

            with torch.no_grad():
                gat_output = gat_model(subg.x, subg.edge_index)
                combined_output = torch.cat((gat_output, word2vec_tensor), dim=1)
                tgn_output = tgn_model(combined_output, subg.edge_index, subg.edge_time)

          
                final_output = att_model(tgn_output.unsqueeze(1))  # [B, 1, C]
                final_output_np = final_output.squeeze(1).detach().cpu().numpy()

                xgb_preds = xgb_model.predict(final_output_np)
                xgb_probs = xgb_model.predict_proba(final_output_np)
                conf = np.max(xgb_probs, axis=1)

                pred = torch.tensor(xgb_preds, dtype=torch.long).to(device)
                conf = torch.tensor(conf).to(device)
                true_label = subg.y.to(device)

                cond = (pred == true_label)
                flag[subg.n_id[cond]] = False  


    index = utils.mask_to_index(flag).tolist()
    ids = set([mapp[i] for i in index])
    alerts = helper(set(ids), all_ids, GT_mal, edges, mapp)


if __name__ == "__main__":
    main()
