
# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import re
import shutil
from tqdm import tqdm
import networkx as nx
import pickle as pkl


def extract_uuid(line):
    pattern_uuid = re.compile(r'uuid\":\"(.*?)\"')
    return pattern_uuid.findall(line)


def extract_subject_type(line):
    pattern_type = re.compile(r'type\":\"(.*?)\"')
    return pattern_type.findall(line)


def show(file_path):
    print(f"Processing {file_path}")


def extract_edge_info(line):
    pattern_src = re.compile(r'subject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
    pattern_dst1 = re.compile(r'predicateObject\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
    pattern_dst2 = re.compile(r'predicateObject2\":{\"com.bbn.tc.schema.avro.cdm18.UUID\":\"(.*?)\"}')
    pattern_type = re.compile(r'type\":\"(.*?)\"')
    pattern_time = re.compile(r'timestampNanos\":(.*?),')

    edge_type = extract_subject_type(line)[0]
    timestamp = pattern_time.findall(line)[0]
    src_id = pattern_src.findall(line)

    if len(src_id) == 0:
        return None, None, None, None, None

    src_id = src_id[0]
    dst_id1 = pattern_dst1.findall(line)
    dst_id2 = pattern_dst2.findall(line)

    if len(dst_id1) > 0 and dst_id1[0] != 'null':
        dst_id1 = dst_id1[0]
    else:
        dst_id1 = None

    if len(dst_id2) > 0 and dst_id2[0] != 'null':
        dst_id2 = dst_id2[0]
    else:
        dst_id2 = None

    return src_id, edge_type, timestamp, dst_id1, dst_id2


def process_data(file_path):
    id_nodetype_map = {}
    notice_num = 1000000
    for i in range(100):
        now_path = file_path + '.' + str(i)
        if i == 0:
            now_path = file_path
        if not os.path.exists(now_path):
            break

        with open(now_path, 'r', encoding='utf-8') as f:
            show(now_path)
            cnt = 0
            for line in f:
                cnt += 1
                if cnt % notice_num == 0:
                    print(cnt)

                if 'com.bbn.tc.schema.avro.cdm18.Event' in line or 'com.bbn.tc.schema.avro.cdm18.Host' in line:
                    continue

                if 'com.bbn.tc.schema.avro.cdm18.TimeMarker' in line or 'com.bbn.tc.schema.avro.cdm18.StartMarker' in line:
                    continue

                if 'com.bbn.tc.schema.avro.cdm18.UnitDependency' in line or 'com.bbn.tc.schema.avro.cdm18.EndMarker' in line:
                    continue

                uuid = extract_uuid(line)[0]
                subject_type = extract_subject_type(line)

                if len(subject_type) < 1:
                    if 'com.bbn.tc.schema.avro.cdm18.MemoryObject' in line:
                        id_nodetype_map[uuid] = 'MemoryObject'
                        continue
                    if 'com.bbn.tc.schema.avro.cdm18.NetFlowObject' in line:
                        id_nodetype_map[uuid] = 'NetFlowObject'
                        continue
                    if 'com.bbn.tc.schema.avro.cdm18.UnnamedPipeObject' in line:
                        id_nodetype_map[uuid] = 'UnnamedPipeObject'
                        continue

                id_nodetype_map[uuid] = subject_type[0]

    return id_nodetype_map


def process_edges(file_path, id_nodetype_map):
    notice_num = 1000000
    not_in_cnt = 0

    for i in range(100):
        now_path = file_path + '.' + str(i)
        if i == 0:
            now_path = file_path
        if not os.path.exists(now_path):
            break

        with open(now_path, 'r', encoding='utf-8') as f, open(now_path+'.txt', 'w', encoding='utf-8') as fw:
            cnt = 0
            for line in f:
                cnt += 1
                if cnt % notice_num == 0:
                    print(cnt)

                if 'com.bbn.tc.schema.avro.cdm18.Event' in line:
                    src_id, edge_type, timestamp, dst_id1, dst_id2 = extract_edge_info(line)

                    if src_id is None or src_id not in id_nodetype_map:
                        not_in_cnt += 1
                        continue

                    src_type = id_nodetype_map[src_id]

                    if dst_id1 is not None and dst_id1 in id_nodetype_map:
                        dst_type1 = id_nodetype_map[dst_id1]
                        this_edge1 = f"{src_id}\t{src_type}\t{dst_id1}\t{dst_type1}\t{edge_type}\t{timestamp}\n"
                        fw.write(this_edge1)

                    if dst_id2 is not None and dst_id2 in id_nodetype_map:
                        dst_type2 = id_nodetype_map[dst_id2]
                        this_edge2 = f"{src_id}\t{src_type}\t{dst_id2}\t{dst_type2}\t{edge_type}\t{timestamp}\n"
                        fw.write(this_edge2)


def run_data_processing(dataset):
    if dataset == 'trace':
        # os.system('tar -zxvf ta1-trace-e3-official-1.json.tar.gz')
        path_list = ['ta1-trace-e3-official-1.json']
    elif dataset == 'theia':
        # os.system('tar -zxvf ta1-theia-e3-official-1r.json.tar.gz')
        # os.system('tar -zxvf ta1-theia-e3-official-6r.json.tar.gz')
        path_list = ['ta1-theia-e3-official-1r.json', 'ta1-theia-e3-official-6r.json']
    elif dataset == 'cadets':
        # os.system('tar -zxvf ta1-cadets-e3-official.json.tar.gz')
        # os.system('tar -zxvf ta1-cadets-e3-official-2.json.tar.gz')
        path_list = ['ta1-cadets-e3-official.json', 'ta1-cadets-e3-official-2.json']
    else:
        print("Unsupported dataset.")
        return
    base_path = dataset

    for path in path_list:
        id_nodetype_map = process_data("./"+base_path+"/"+path)
        process_edges("./"+base_path+"/"+path, id_nodetype_map)

    if dataset == 'trace':
        shutil.copy("./"+base_path+"/"+'ta1-trace-e3-official-1.json.txt', "./"+base_path+"/"+'trace_train.txt')
        shutil.copy("./"+base_path+"/"+'ta1-trace-e3-official-1.json.4.txt', "./"+base_path+"/"+'trace_test.txt')
    elif dataset == 'theia':
        shutil.copy("./"+base_path+"/"+'ta1-theia-e3-official-1r.json.txt', "./"+base_path+"/"+'theia_train.txt')
        shutil.copy("./"+base_path+"/"+'ta1-theia-e3-official-6r.json.8.txt', "./"+base_path+"/"+'theia_test.txt')
    elif dataset == 'cadets':
        shutil.copy("./"+base_path+"/"+'ta1-cadets-e3-official.json.1.txt', "./"+base_path+"/"+'cadets_train.txt')
        shutil.copy("./"+base_path+"/"+'ta1-cadets-e3-official-2.json.txt', "./"+base_path+"/"+'cadets_test.txt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDM Parser')
    parser.add_argument("--dataset", type=str, default="cadets")
    args = parser.parse_args()

    if args.dataset not in ['trace', 'theia', 'cadets']:
        raise ValueError(f"Unsupported dataset: '{args.dataset}'. Supported options are: 'trace', 'theia', 'cadets'.")

    run_data_processing(args.dataset)


