import os

import joblib
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch.nn import CrossEntropyLoss, Linear
from sklearn.utils import class_weight
from tqdm import tqdm
from gensim.models import Word2Vec
from torch_geometric import utils

from model.PositionalEncoder import PositionalEncoder
from model.TGN import TemporalGraphNetwork
from model.GAT import GAT
from model.MultiHeadSelfAttention import MultiHeadSelfAttentionModel, MultiHeadSelfAttention
from utils.graphutils_trace import add_attributes, prepare_graph
from utils.word2vec_utils import train_word2vec


encoder = PositionalEncoder(30)


# 在此处定义 infer_word2vec_embedding 函数
def infer_word2vec_embedding(document, w2vmodel):
    word_embeddings = [w2vmodel.wv[word] for word in document if word in w2vmodel.wv]
    if not word_embeddings:
        return np.zeros(30)  # 修改为 30 与 PositionalEncoder 的 d_model 一致
    output_embedding = np.array(word_embeddings)
    output_embedding = torch.tensor(output_embedding, dtype=torch.float)
    output_embedding = encoder.embed(output_embedding)
    output_embedding = output_embedding.detach().cpu().numpy()
    return np.mean(output_embedding, axis=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load and process data
    with open("../data/trace/trace_train.txt") as f:
        data = f.read().split('\n')
    data = [line.split('\t') for line in data if line]
    df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
    df = df.dropna()
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    print(f"DataFrame Shape: {df.shape}")

    df = add_attributes(df, "../data/trace/ta1-trace-e3-official-1.json.1")
    print(f"Attributes added")
    phrases, labels, edges, mapp, node_types, edges_attr = prepare_graph(df)
    print(f"Phrases, Labels, Edges, and Mapping prepared")
    model_path = "word2vec_trace.model"
    if os.path.exists(model_path):
        w2vmodel = Word2Vec.load(model_path)
        print("Word2Vec model loaded.")
    else:
        phrases = phrases  # 在这里定义你的训练数据
        dataset_name = "trace"  # 根据需要设置数据集名称
        w2vmodel = train_word2vec(phrases, dataset_name, vector_size=30, window=5, min_count=1, workers=8, epochs=300)
        print("Word2Vec model trained and saved.")
    all_ids = list(df['actorID']) + list(df['objectID'])
    all_ids = set(all_ids)
    num_classes = 12
    gat_model = GAT(in_channel=12, out_channel=30).to(device)
    tgn_model = TemporalGraphNetwork(in_channels=60, out_channels=32).to(device)
    model = MultiHeadSelfAttentionModel(in_channels=32, out_channels=12, num_classes=num_classes,
                                        num_heads=4).to(device)
    optimizer = optim.Adam(
        list(model.parameters()) + list(tgn_model.parameters()) + list(gat_model.parameters()),
        lr=0.0005, weight_decay=5e-4
    )
    class_weights = class_weight.compute_class_weight(class_weight=None, classes=np.arange(num_classes), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = CrossEntropyLoss(weight=class_weights, reduction='mean')
    graph = Data(
        x=torch.tensor(np.array(node_types), dtype=torch.float).to(device),
        y=torch.tensor(labels, dtype=torch.long).to(device),
        edge_index=torch.tensor(edges, dtype=torch.long).to(device),
        edge_time=torch.tensor(edges_attr, dtype=torch.float).to(device)
    )

    graph.n_id = torch.arange(graph.num_nodes)
    mask = torch.tensor([True] * graph.num_nodes, dtype=torch.bool)

    # Training model
    for epoch in range(50):
        print(f"Epoch: {epoch}")
        all_final_outputs = []
        all_labels = []
        loader = NeighborLoader(graph, num_neighbors=[15, 10], batch_size=26000, input_nodes=mask)
        total_loss = 0
        for subg in tqdm(loader, desc="Training", leave=False):
            # 获取当前子图中的短语或节点ID
            subg = subg.to(device)
            optimizer.zero_grad()
            # 获取当前子图中的短语或节点ID
            current_phrases = [phrases[idx] for idx in subg.n_id.tolist()]
            # 假设 infer_word2vec_embedding(p) 返回的是 numpy.ndarray
            word2vec_embeddings = [infer_word2vec_embedding(p, w2vmodel) for p in current_phrases]

            # 将列表转换为 numpy 数组
            word2vec_np_array = np.array(word2vec_embeddings)

            # 将 numpy 数组转换为 PyTorch 张量
            word2vec_output = torch.tensor(word2vec_np_array, dtype=torch.float).to(device)
            # Process node features with GAT
            gat_output = gat_model(subg.x, subg.edge_index)
            combined_output = torch.cat((gat_output, word2vec_output), dim=1)
            # Process GAT output with TGN
            tgn_output = tgn_model(combined_output, subg.edge_index, subg.edge_time)
            tgn_output = tgn_output.unsqueeze(1)
            # Self-Attention output
            final_output = model(tgn_output)
            loss = criterion(final_output.view(-1, num_classes), subg.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * subg.batch_size
            all_final_outputs.append(final_output.detach().cpu().numpy())
            all_labels.append(subg.y.cpu().numpy())
            del subg
            torch.cuda.empty_cache()


        loader = NeighborLoader(graph, num_neighbors=[15, 10], batch_size=26000, input_nodes=mask)
        for subg in tqdm(loader, desc="Evaluation", leave=False):
            subg = subg.to(device)
            gat_model.eval()
            tgn_model.eval()
            model.eval()
            with torch.no_grad():
                # 获取当前子图中的短语或节点ID
                current_phrases = [phrases[idx] for idx in subg.n_id.tolist()]
                # 假设 infer_word2vec_embedding(p) 返回的是 numpy.ndarray
                word2vec_embeddings = [infer_word2vec_embedding(p, w2vmodel) for p in current_phrases]

                # 将列表转换为 numpy 数组
                word2vec_np_array = np.array(word2vec_embeddings)

                # 将 numpy 数组转换为 PyTorch 张量
                word2vec_output = torch.tensor(word2vec_np_array, dtype=torch.float).to(device)
                # Process node features with GAT
                gat_output = gat_model(subg.x, subg.edge_index)
                combined_output = torch.cat((gat_output, word2vec_output), dim=1)
                # Process GAT output with TGN
                tgn_output = tgn_model(combined_output, subg.edge_index, subg.edge_time)
                tgn_output = tgn_output.unsqueeze(1)
                # Self-Attention output
                final_output = model(tgn_output)

                sorted, indices = final_output.view(-1, num_classes).sort(dim=1, descending=True)
                conf = (sorted[:, 0] - sorted[:, 1]) / sorted[:, 0]
                conf = (conf - conf.min()) / conf.max()
                pred = indices[:, 0]


                cond = (pred == subg.y) & (conf >= 0.8)
                mask[subg.n_id[cond]] = False
        torch.save(gat_model.state_dict(),
                   f'../trained_weights/trace/gat{epoch}.pth')
        torch.save(tgn_model.state_dict(),
                   f'../trained_weights/trace/tgn{epoch}.pth')
        torch.save(model.state_dict(),
                   f'../trained_weights/trace/selfattention{epoch}.pth')

    print("down")

if __name__ == "__main__":
    main()
