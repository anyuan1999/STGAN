import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import joblib
from tqdm import tqdm
from gensim.models import Word2Vec

from model.PositionalEncoder import PositionalEncoder
from model.TGN import TemporalGraphNetwork
from model.GAT import GAT
from model.MultiHeadSelfAttention import MultiHeadSelfAttentionModel
from utils.graphutils_trace import add_attributes, prepare_graph
from utils.word2vec_utils import train_word2vec

# 设置全局编码器
encoder = PositionalEncoder(30)

def infer_word2vec_embedding(document, w2vmodel):
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
    print(f"Using device: {device}")

    # === Step 1: Load and preprocess data ===
    with open("../data/trace/trace_train.txt") as f:
        data = f.read().split('\n')
    data = [line.split('\t') for line in data if line]
    df = pd.DataFrame(data, columns=['actorID', 'actor_type', 'objectID', 'object', 'action', 'timestamp'])
    df = df.dropna()
    df.sort_values(by='timestamp', ascending=True, inplace=True)

    df = add_attributes(df, "../data/trace/ta1-trace-e3-official-1.json.1")
    phrases, labels, edges, mapp, node_types, edges_attr = prepare_graph(df)


    model_path = "word2vec_trace.model"
    if os.path.exists(model_path):
        w2vmodel = Word2Vec.load(model_path)
    else:
        w2vmodel = train_word2vec(phrases, "trace", vector_size=30, window=5, min_count=1, workers=8, epochs=300)


    graph = Data(
        x=torch.tensor(np.array(node_types), dtype=torch.float).to(device),
        y=torch.tensor(labels, dtype=torch.long).to(device),
        edge_index=torch.tensor(edges, dtype=torch.long).to(device),
        edge_time=torch.tensor(edges_attr, dtype=torch.float).to(device)
    )
    graph.n_id = torch.arange(graph.num_nodes)
    mask = torch.tensor([True] * graph.num_nodes, dtype=torch.bool)


    num_classes = 12
    gat_model = GAT(in_channel=12, out_channel=30).to(device)
    tgn_model = TemporalGraphNetwork(in_channels=60, out_channels=32).to(device)
    attention_model = MultiHeadSelfAttentionModel(in_channels=32, out_channels=12, num_classes=num_classes, num_heads=4).to(device)

    optimizer = optim.Adam(
        list(gat_model.parameters()) + list(tgn_model.parameters()) + list(attention_model.parameters()),
        lr=0.001, weight_decay=5e-4
    )

    class_weights = class_weight.compute_class_weight(class_weight=None, classes=np.arange(num_classes), y=labels)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))

  
    for epoch in range(30):
        gat_model.train()
        tgn_model.train()
        attention_model.train()
        loader = NeighborLoader(graph, num_neighbors=[15, 10], batch_size=26000, input_nodes=mask)
        total_loss = 0

        for subg in tqdm(loader, desc=f"Epoch {epoch} Training", leave=False):
            subg = subg.to(device)
            optimizer.zero_grad()
            current_phrases = [phrases[idx] for idx in subg.n_id.tolist()]
            word2vec_embeddings = [infer_word2vec_embedding(p, w2vmodel) for p in current_phrases]
            word2vec_tensor = torch.tensor(np.array(word2vec_embeddings), dtype=torch.float).to(device)

            gat_output = gat_model(subg.x, subg.edge_index)
            combined_output = torch.cat((gat_output, word2vec_tensor), dim=1)
            tgn_output = tgn_model(combined_output, subg.edge_index, subg.edge_time)
            final_output = attention_model(tgn_output.unsqueeze(1))

            loss = criterion(final_output.view(-1, num_classes), subg.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch} loss: {total_loss:.4f}")


    gat_model.eval()
    tgn_model.eval()
    with torch.no_grad():
        gat_output = gat_model(graph.x, graph.edge_index)
        word2vec_embeddings = [infer_word2vec_embedding(p, w2vmodel) for p in phrases]
        word2vec_tensor = torch.tensor(np.array(word2vec_embeddings), dtype=torch.float).to(device)
        combined_output = torch.cat((gat_output, word2vec_tensor), dim=1)
        tgn_output = tgn_model(combined_output, graph.edge_index, graph.edge_time)


    print("Training XGBoost classifier...")
    final_output = attention_model(tgn_output.unsqueeze(1))
    features = final_output.squeeze(1).detach().cpu().numpy()
    labels_np = graph.y.cpu().numpy()

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb_model.fit(features, labels_np)


    preds = xgb_model.predict(features)
    print("\nClassification Report:\n")
    print(classification_report(labels_np, preds, digits=4))


    torch.save(gat_model.state_dict(), '../trained_weights/trace_xgb/gat_final.pth')
    torch.save(tgn_model.state_dict(), '../trained_weights/trace_xgb/tgn_final.pth')
    torch.save(attention_model.state_dict(), '../trained_weights/trace_xgb/selfattention_final.pth')
    joblib.dump(xgb_model, '../trained_weights/trace_xgb/xgb_model.pkl')
    print("All models saved to ../trained_weights/trace/")

if __name__ == "__main__":
    main()
