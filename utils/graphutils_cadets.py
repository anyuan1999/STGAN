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
        'SUBJECT_PROCESS': 0, 'FILE_OBJECT_FILE': 1, 'FILE_OBJECT_UNIX_SOCKET': 2,
        'UnnamedPipeObject': 3, 'NetFlowObject': 4, 'FILE_OBJECT_DIR': 5
    }

    num_classes = len(dummies)  # 确定类别总数，用于生成 one-hot 编码

    # 遍历数据框中的每一行，构建节点属性和边
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
    f = open(p)
    data = [json.loads(x) for x in f if "EVENT" in x]

    info = []
    for x in data:
        try:
            action = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['type']
        except:
            action = ''
        try:
            actor = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['subject']['com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            actor = ''
        try:
            obj = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject'][
                'com.bbn.tc.schema.avro.cdm18.UUID']
        except:
            obj = ''
        try:
            timestamp = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['timestampNanos']
        except:
            timestamp = ''
        try:
            cmd = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['properties']['map']['exec']
        except:
            cmd = ''
        try:
            path = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObjectPath']['string']
        except:
            path = ''
        try:
            path2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2Path']['string']
        except:
            path2 = ''
        try:
            obj2 = x['datum']['com.bbn.tc.schema.avro.cdm18.Event']['predicateObject2'][
                'com.bbn.tc.schema.avro.cdm18.UUID']
            info.append({'actorID': actor, 'objectID': obj2, 'action': action, 'timestamp': timestamp, 'exec': cmd,
                         'path': path2})
        except:
            pass

        info.append(
            {'actorID': actor, 'objectID': obj, 'action': action, 'timestamp': timestamp, 'exec': cmd, 'path': path})

    rdf = pd.DataFrame.from_records(info).astype(str)
    d = d.astype(str)

    return d.merge(rdf, how='inner', on=['actorID', 'objectID', 'action', 'timestamp']).drop_duplicates()
