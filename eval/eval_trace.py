import os
import pandas as pd
import torch
import torch.nn as nn
import json
import pickle
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from gensim.models import Word2Vec
from torch_geometric import utils
from ..utils.helper import helper
from ..model.MultiHeadSelfAttention import MultiHeadSelfAttentionModel,MultiHeadSelfAttention
from ..model.GAT import GAT
from ..utils.graphutils_trace import add_attributes,prepare_graph
from ..model.PositionalEncoder import PositionalEncoder
from ..model.TGN import TemporalGraphNetwork


# 加载全局的 Word2Vec 模型和编码器
w2vmodel = Word2Vec.load("word2vec_trace_E3.model")
encoder = PositionalEncoder(30)

def infer_word2vec_embedding(document):
    word_embeddings = [w2vmodel.wv[word] for word in document if word in w2vmodel.wv]
    if not word_embeddings:
        return np.zeros(30)  # 修改为 15 与 PositionalEncoder 的 d_model 一致
    output_embedding = np.array(word_embeddings)  # 先将列表转换为 numpy 数组
    output_embedding = torch.tensor(output_embedding, dtype=torch.float)  # 然后再转换为 PyTorch 张量
    output_embedding = encoder.embed(output_embedding)  # 使用 embed 方法
    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load and process data
    with open("../../../data_raw/trace/trace_test.txt") as f:
        data = f.read().split('\n')
    data = [line.split('\t') for line in data if line]
    df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
    df = df.dropna()
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    df = add_attributes(df, "../../../data_raw/trace/ta1-trace-e3-official-1.json.4")
    with open("../../../data_files/trace.json", "r") as json_file:
        GT_mal = set(json.load(json_file))
    phrases, labels, edges, mapp, node_types, edges_attr = prepare_graph(df)
    all_ids = list(df['actorID']) + list(df['objectID'])
    all_ids = set(all_ids)
    gat_model = GAT(in_channel=12, out_channel=30).to(device)
    tgn_model = TemporalGraphNetwork(in_channels=60, out_channels=32).to(device)
    num_classes = 12
    model = MultiHeadSelfAttentionModel(in_channels=32, out_channels=12, num_classes=num_classes, num_heads=4).to(device)
    graph = Data(x=torch.tensor(node_types, dtype=torch.float).to(device),
                 y=torch.tensor(labels, dtype=torch.long).to(device),
                 edge_index=torch.tensor(edges, dtype=torch.long).to(device),
                 edge_time=torch.tensor(edges_attr, dtype=torch.float).to(device))
    graph.n_id = torch.arange(graph.num_nodes).to(device)  # 确保 n_id 在正确的设备上
    flag = torch.tensor([True] * graph.num_nodes, dtype=torch.bool).to(device)

    print(f"Graph Created")
    for m_n in range(100):
        print(f"Epoch: {m_n}")
        gat_model.load_state_dict(
            torch.load(
                f'../trained_weights/trace/gat{m_n}.pth',
                map_location=device))
        gat_model.eval()
        tgn_model.load_state_dict(
            torch.load(
                f'../trained_weights/trace/tgn{m_n}.pth',
                map_location=device))
        tgn_model.eval()
        model.load_state_dict(
            torch.load(
                f'../trained_weights/trace/selfattention{m_n}.pth',
                map_location=device))
        model.eval()
        # 初始化标签列表

        loader = NeighborLoader(graph, num_neighbors=[-1, -1], batch_size=26000)
        for subg in tqdm(loader, desc="Evaluation", leave=False):
            subg.x = subg.x.to(device)
            subg.edge_index = subg.edge_index.to(device)
            subg.y = subg.y.to(device)
            subg.edge_time = subg.edge_time.to(device)
            with torch.no_grad():
                current_phrases = [phrases[idx] for idx in subg.n_id.tolist()]
                # 假设 infer_word2vec_embedding(p) 返回的是 numpy.ndarray
                word2vec_embeddings = [infer_word2vec_embedding(p) for p in current_phrases]

                # 将列表转换为 numpy 数组
                word2vec_np_array = np.array(word2vec_embeddings)

                # 将 numpy 数组转换为 PyTorch 张量
                word2vec_output = torch.tensor(word2vec_np_array, dtype=torch.float).to(device)
                # 使用 GAT 处理节点特征
                gat_output = gat_model(subg.x, subg.edge_index)
                combined_output = torch.cat((gat_output, word2vec_output), dim=1)
                # 使用 TGN 处理 GAT 输出
                tgn_output = tgn_model(combined_output, subg.edge_index, subg.edge_time)
                tgn_output = tgn_output.unsqueeze(1)
                # Self-Attention output
                final_output = model(tgn_output)
                sorted, indices = final_output.view(-1, num_classes).sort(dim=1, descending=True)
                conf = (sorted[:, 0] - sorted[:, 1]) / sorted[:, 0]
                conf = (conf - conf.min()) / conf.max()
                pred = indices[:, 0]
                cond = (pred == subg.y)
                flag[subg.n_id[cond]] = torch.logical_and(flag[subg.n_id[cond]],
                                                          torch.tensor([False] * len(flag[subg.n_id[cond]]),
                                                                       dtype=torch.bool).to(device))
    index = utils.mask_to_index(flag).tolist()
    ids = set([mapp[x] for x in index])
    alerts = helper(set(ids), set(all_ids), GT_mal, edges, mapp)


if __name__ == "__main__":
    main()
