import torch
import json
import pandas as pd
import numpy as np


def add_node_properties(nodes, node_id, properties):
    if node_id not in nodes:
        nodes[node_id] = []
    nodes[node_id].extend(properties)


def update_edge_index(edges, edge_index, index):
    for src_id, dst_id in edges:
        src = index[src_id]
        dst = index[dst_id]
        edge_index[0].append(src)
        edge_index[1].append(dst)


def prepare_graph(df):
    nodes, labels, edges, edge_attr = {}, {}, [], []
    dummies = {
        "SUBJECT_PROCESS": 0, "MemoryObject": 1, "FILE_OBJECT_BLOCK": 2,
        "NetFlowObject": 3, "PRINCIPAL_REMOTE": 4, "PRINCIPAL_LOCAL": 5
    }

    num_classes = len(dummies)  # 确定类别总数，用于生成 one-hot 编码

    for _, row in df.iterrows():
        action = row["action"]
        properties = [row['exec'], action] + ([row['path']] if row['path'] else [])

        actor_id = row["actorID"]
        add_node_properties(nodes, actor_id, properties)
        labels[actor_id] = dummies[row['actor_type']]

        object_id = row["objectID"]
        add_node_properties(nodes, object_id, properties)
        labels[object_id] = dummies[row['object']]

        edges.append((actor_id, object_id))
        edge_attr.append(float(row["timestamp"]))  # 添加时间戳作为边属性

    features, feat_labels, edge_index, index_map = [], [], [[], []], {}
    one_hot_labels = []

    for node_id, props in nodes.items():
        features.append(props)
        feat_labels.append(labels[node_id])
        index_map[node_id] = len(features) - 1

        # 生成 one-hot 编码向量
        one_hot_vector = np.zeros(num_classes)
        one_hot_vector[labels[node_id]] = 1
        one_hot_labels.append(one_hot_vector)

    # 更新边索引
    update_edge_index(edges, edge_index, index_map)

    return features, feat_labels, edge_index, list(index_map.keys()), one_hot_labels, edge_attr


def add_attributes(d, p):
    # 使用 utf-8 编码打开文件，忽略解码错误
    with open(p, 'r', encoding='utf-8', errors='ignore') as f:
        data = []
        for line in f:
            if "EVENT" in line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    # 处理 JSON 解码错误
                    print(f"Skipping invalid JSON line: {line}")

    info = []
    for x in data:
        datum = x.get('datum', {}).get('com.bbn.tc.schema.avro.cdm18.Event', {})
        if datum is None:
            # 如果 datum 为 None，则跳过此条数据
            print(f"Skipping entry with missing 'datum': {x}")
            continue

        action = datum.get('type', '')
        actor = datum.get('subject', {}).get('com.bbn.tc.schema.avro.cdm18.UUID', '')
        obj = datum.get('predicateObject', {}).get('com.bbn.tc.schema.avro.cdm18.UUID', '')
        timestamp = datum.get('timestampNanos', '')
        cmd = datum.get('properties', {}).get('map', {}).get('cmdLine', '')

        path = datum.get('predicateObjectPath', None)
        path = path.get('string', '') if path is not None else ''

        path2 = datum.get('predicateObject2Path', None)
        path2 = path2.get('string', '') if path2 is not None else ''

        obj2 = datum.get('predicateObject2', {}).get('com.bbn.tc.schema.avro.cdm18.UUID', '')

        if obj2:
            info.append({'actorID': actor, 'objectID': obj2, 'action': action, 'timestamp': timestamp, 'exec': cmd,
                         'path': path2})

        info.append(
            {'actorID': actor, 'objectID': obj, 'action': action, 'timestamp': timestamp, 'exec': cmd, 'path': path})

    rdf = pd.DataFrame.from_records(info).astype(str)
    d = d.astype(str)

    # Merge dataframes and drop duplicates
    merged_df = d.merge(rdf, how='inner', on=['actorID', 'objectID', 'action', 'timestamp']).drop_duplicates()

    return merged_df




