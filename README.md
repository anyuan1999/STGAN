# STGAN

Main code for STGAN∗: Detecting Host Threats via Spatial Temporal Graph Attention Network in Host Provenance Graphs



## Overview

The overall workflow of STGAN is illustrated in above figure. STGAN accepts streaming audit log input and slices the log information into segments. For each subgraph within a segment, STGAN performs both spatial and temporal embedding. For spatial embedding, STGAN first constructs sentences based on the first-hop neighbors of each node and uses Word2Vec to learn the semantic features of the nodes. These semantic features are then used as the initial node embeddings, and GAT is applied to extract structural features, thereby forming a comprehensive spatial feature representation. For temporal embedding, STGAN employs TGN to capture the temporal features of the nodes. Finally, STGAN uses a multi-head self-attention mechanism to fuse the spatial and temporal features, generating a complete spatial-temporal graph embedding to support anomaly detection tasks. In the anomaly detection phase, STGAN utilizes XGBoost as the anomaly detector to perform anomaly detection.

## Directory Structure

- **Utils**: Contains necessary tools.
- **data**: Includes datasets TRACE, CADETS, THEIA, and data processing tools.
- **eval**: Contains evaluation scripts.
- **model**: Houses graph embedding models.
- **train**: Contains training scripts for all datasets.

## Dataset Links

- DRAPA TC: [Google Drive](https://drive.google.com/open?id=1QlbUFWAGq3Hpl8wVdzOdIoZLFxkII4EK)

## Dependencies

- Python 3.9
- torch==1.12.0+cu113
- torch-geometric==2.1.0
- torch-cluster==1.6.0+pt112cu113
- torch-scatter==2.0.9
- torch-sparse==0.6.15+pt112cu113
- torch-spline-conv==1.2.1+pt112cu113
- DGL 1.0.0
- scikit-learn==1.1.1
- xgboost==0.90
- gensim==4.3.0
- networkx==3.0

More content and features will be open-sourced in future updates.
