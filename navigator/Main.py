import os
import yaml
import torch
import json
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm
import networkx as nx
import torch.nn.functional as F
#from torch_geometric.data import Data
from torch.utils.data import DataLoader 
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from transformers import AutoTokenizer, AutoConfig, AdamW, AutoModelForCausalLM, AutoModel, LlamaTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from Dataset import TrainDataset, NodeDataset, TestEdgeDataset
from Dataset_path import TrainDataset_p, NodeDataset_p, TestEdgeDataset_p
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training
)

from Model import Bert, Bert_v2, Bert_v3, Bert_v4, Bert_v5

def get_prompt(key, id_2_text_dict):
    title, abstract = id_2_text_dict[key]
    if title == None:
        title = 'None'
    if abstract == None:
        abstract = 'None'

    return title + abstract

def read_yaml_file(file_path):
        with open(file_path, 'r') as file:
            try:
                data = yaml.safe_load(file)
                return data
            except yaml.YAMLError as e:
                print(f"Error reading YAML file: {e}")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

'''
def translate_graph(graph_data, test_data):
    print("translating raw graph")
    raw_graph = graph_data[0]
    raw_id_2_tile_abs = graph_data[2]
    raw_id_pair_2_sentence = graph_data[3]

    old_id_2_new_id = dict()
    new_id_2_title_abs = dict()
    new_id_pair_2_sentence = dict()
    count = 0
    
    new_graph = nx.Graph()
    new_test_data = []

    # translate nodes
    for node in tqdm(sorted(raw_graph.nodes())):
        old_id_2_new_id[node] = count
        new_id_2_title_abs[count] = raw_id_2_tile_abs[node]
        count += 1
    
    # translate edges
    for edge in tqdm(raw_graph.edges()):
        new_edge = (old_id_2_new_id[edge[0]], old_id_2_new_id[edge[1]])
        new_graph.add_edge(new_edge[0], new_edge[1])
        try:
            new_id_pair_2_sentence[new_edge] = raw_id_pair_2_sentence[edge]
        except:
            new_id_pair_2_sentence[new_edge] = raw_id_pair_2_sentence[(edge[1], edge[0])]
    
    # translate test samples
    for sample in tqdm(test_data): # pos
        new_source_id = old_id_2_new_id[sample[0]]
        new_target_id = old_id_2_new_id[sample[1]]

        if new_graph.has_edge(new_source_id, new_target_id) == False and new_graph.has_edge(new_target_id, new_source_id) == False:
            print("pos_error")
        
        new_source_edge = []
        for edge in sample[2]:
            new_source_edge.append((old_id_2_new_id[edge[0]], old_id_2_new_id[edge[1]]))
        
        new_target_edge = []
        for edge in sample[3]:
            new_target_edge.append((old_id_2_new_id[edge[0]], old_id_2_new_id[edge[1]]))
        new_test_data.append([new_source_id, new_target_id, new_source_edge, new_target_edge, 1])
    
    return [new_graph, new_id_2_title_abs, new_id_pair_2_sentence], new_test_data
'''

def get_rank_metric(s_ind, t_ind, query_embs, all_node_embeddings, model, train_graph):
    # get score of candidate samples
    R_list, RR_list, H1_list, H3_list, H10_list = [], [], [], [], []
    for i in range(query_embs.size()[0]):
        s_embeddings = query_embs[s_ind[i]]
        tmp_scores = model.Cos(s_embeddings, all_node_embeddings).flatten()

        #tmp_s_embeddings = s_embeddings[i].reshape(1, -1).repeat(all_node_embeddings.size()[0], 1)
        #tmp_scores = - model.TransE(tmp_s_embeddings, all_node_embeddings).flatten()
        
        target_score = tmp_scores[int(t_ind[i])] # 1

        # repalce valid candidates
        s_neighbors = list(nx.all_neighbors(train_graph, int(s_ind[i])))
        tmp_scores[s_neighbors] = target_score.clone()
        
        tmp_list = target_score - tmp_scores
        rank = len(tmp_list[tmp_list < 0]) + 1
        
        R_list.append(rank)
        RR_list.append(1.0/float(rank))
        
        if rank <= 1:
            H1_list.append(1)
        else:
            H1_list.append(0)
        
        if rank <= 3:
            H3_list.append(1)
        else:
            H3_list.append(0)
        
        if rank <= 10:
            H10_list.append(1)
        else:
            H10_list.append(0)
    
    return RR_list, H1_list, H3_list, H10_list


def get_precision_metric(s_ind, query_embs, all_node_embeddings, model, truth, all_nodes, all_nei_embeddings=None):
    # get score of candidate samples
    R_list, RR_list, H1_list, H3_list, H10_list = [], [], [], [], []
    top_10_result, top_5_result, top_1_result = [], [], []
    all_nodes = np.array(all_nodes)
    query_ind_list = []
    top10_list = []
    candidate_list = []
    for i in range(query_embs.size()[0]):
        query_ind = s_ind[i]
        s_embeddings = query_embs[i]
        tmp_scores = model.Cos(s_embeddings, all_node_embeddings).flatten()
        #if all_nei_embeddings == None:
        #    tmp_scores = model.Cos(s_embeddings, all_node_embeddings).flatten()
        #else:
        #    tmp_scores = model.nei_score(s_embeddings, all_node_embeddings, all_nei_embeddings).flatten()
        sorted_scores, idxs = torch.sort(tmp_scores, descending = True)
        top_10 = all_nodes[[int(k) for k in idxs[:10]]]
        top_5 = all_nodes[[int(k) for k in idxs[:5]]]
        top_1 = all_nodes[[int(k) for k in idxs[:1]]]

        # valid candidates
        candidates = list(truth[int(s_ind[i])])
        top_10_hits = len(set(top_10) & set(candidates))
        top_5_hits = len(set(top_5) & set(candidates))
        top_1_hits = len(set(top_1) & set(candidates))
        
        top_10_result.append(float(top_10_hits)/min(10.0, len(candidates)))
        top_5_result.append(float(top_5_hits)/min(5.0, len(candidates)))
        top_1_result.append(float(top_1_hits)/min(1.0, len(candidates)))

        query_ind_list.append(query_ind)
        top10_list.append(top_10)
        candidate_list.append(candidates)
    
    return np.mean(top_10_result), np.mean(top_5_result), np.mean(top_1_result), query_ind_list, top10_list, candidate_list

def get_precision_metric_sample(s_ind, query_embs, all_node_embeddings, model, truth, all_nodes, all_nei_embeddings=None):
    # get score of candidate samples
    # get score of candidate samples
    R_list, RR_list, H1_list, H3_list, H10_list = [], [], [], [], []
    top_10_result, top_5_result, top_1_result = [], [], []
    all_nodes = np.array(all_nodes)
    query_ind_list = []
    top10_list = []
    candidate_list = []
    for i in range(query_embs.size()[0]):
        sample_neg = list(random.sample(range(len(all_node_embeddings)), 9))
        sample_pos = [list(truth[int(s_ind[i])])[0]]
        sample_nodes = np.array(sample_pos + sample_neg)
        sample_embeddings = all_node_embeddings[sample_nodes]

        query_ind = s_ind[i]
        s_embeddings = query_embs[i]
        tmp_scores = model.Cos(s_embeddings, sample_embeddings).flatten()
        #if all_nei_embeddings == None:
        #    tmp_scores = model.Cos(s_embeddings, all_node_embeddings).flatten()
        #else:
        #    tmp_scores = model.nei_score(s_embeddings, all_node_embeddings, all_nei_embeddings).flatten()
        sorted_scores, idxs = torch.sort(tmp_scores, descending = True)
        top_10 = sample_nodes[[int(k) for k in idxs[:10]]]
        top_5 = sample_nodes[[int(k) for k in idxs[:5]]]
        top_1 = sample_nodes[[int(k) for k in idxs[:1]]]

        # valid candidates
        candidates = list(truth[int(s_ind[i])])
        top_10_hits = len(set(top_10) & set(candidates))
        top_5_hits = len(set(top_5) & set(candidates))
        top_1_hits = len(set(top_1) & set(sample_pos))
        
        top_10_result.append(float(top_10_hits)/min(10.0, len(candidates)))
        top_5_result.append(float(top_5_hits)/min(5.0, len(candidates)))
        top_1_result.append(float(top_1_hits)/min(1.0, len(sample_pos)))

        query_ind_list.append(query_ind)
        top10_list.append(top_10)
        candidate_list.append(candidates)
    
    return np.mean(top_10_result), np.mean(top_5_result), np.mean(top_1_result), query_ind_list, top10_list, candidate_list


def retrival_task_eval_v5(model, raw_graph, args, device, tokenizer, all_query_embs, train_graph):
    print("start retrival task eval...")
    if args.train_form == 'link':
        cite_dict_id, id_2_title_abs_test = pickle.load(open("test_set.pkl","rb"))
        Node_dataset = NodeDataset(cite_dict_id, id_2_title_abs_test, args, device, tokenizer)
    if args.train_form == 'path':
        id_2_title_abs_test = raw_graph[2]
        path_data = pickle.load(open('path_data.pkl', 'rb'))
        Node_dataset = NodeDataset_p(raw_graph, path_data, args, device, tokenizer)

    print("generate embeddings for all nodes")
    loader = DataLoader(Node_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    all_node_embeddings = []
    all_nei_embeddings = []
    all_nodes = []
    for i_batch, sample_batched in tqdm(enumerate(loader), disable=False, total=len(loader)):
        batch_input = dict()
        for key in sample_batched.keys():
            batch_input[key] = sample_batched[key].long().cuda()      
        subnode_embeddings, subnei_embeddings = model.generate_candidate_emb(batch_input)
        all_node_embeddings.append(subnode_embeddings.detach())
        all_nei_embeddings.append(subnei_embeddings.detach())
        all_nodes.append(batch_input['target'].detach().cpu())
    all_node_embeddings = torch.cat(all_node_embeddings, 0)
    all_nei_embeddings = torch.cat(all_nei_embeddings, 0)
    all_nodes = torch.cat(all_nodes, 0).squeeze(-1)
    print(all_node_embeddings.size())
    print(all_node_embeddings[0][:10])
    print(all_nei_embeddings[0][:10])

    # get test query embeddings
    
    try:
        test_query_embs = pickle.load(open('test_query_embeddings_'+args.pretrained_model.replace('/','_')+'_'+args.train_form+'.pkl', 'rb'))
        print('---------------------------')
    except:
        # define pre-trained LLM for query embedding
        pretrained_model_name = args.pretrained_model
        config = AutoConfig.from_pretrained(pretrained_model_name)
        hidden_size = config.hidden_size
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        LLM_model = AutoModel.from_pretrained(pretrained_model_name).cuda()
        LLM_model.eval()

        test_query_embs = dict()
        if args.train_form == 'link':
            tmp_list = cite_dict_id.keys()
        if args.train_form == 'path':
            tmp_list = path_data.keys()
        for key in tqdm(tmp_list):
            prompt = get_prompt(key, id_2_title_abs_test)

            encoded_input = tokenizer([prompt], padding = True, truncation=True, max_length=512 , return_tensors='pt')
            with torch.no_grad():
                output = LLM_model(**encoded_input.to('cuda'), output_hidden_states=True).hidden_states[-1]
                sentence_embedding = output[:, 0, :]

            test_query_embs[key] = sentence_embedding[0].cpu()
        pickle.dump(test_query_embs, open('test_query_embeddings_'+args.pretrained_model.replace('/','_')+'_'+args.train_form+'.pkl', 'wb'))
    
    if args.train_form == 'link':
        Test_dataset = TestEdgeDataset(cite_dict_id, id_2_title_abs_test, args, device, tokenizer, test_query_embs)
        truth = cite_dict_id
    if args.train_form == 'path':
        Test_dataset = TestEdgeDataset_p(path_data, args, device, tokenizer, test_query_embs)
        truth = dict()
        for key in path_data.keys():
            truth[key] = []
            paths = path_data[key]
            for p in paths:
                truth[key].append(set(p))
        '''
        for key in path_data.keys():
            tmp_nodes = set()
            paths = path_data[key]
            for p in paths:
                tmp_nodes = tmp_nodes | set(p)
            truth[key] = list(tmp_nodes)
        '''
    print(len(truth[list(truth.keys())[0]]))
    print("test set eval")
    MRR_list, H1_list, H3_list, H10_list = [], [], [], []
    Top_10_list, Top_5_list = [], []
    loader = DataLoader(Test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    for i_batch, sample_batched in tqdm(enumerate(loader), disable=False, total=len(loader)):
        batch_input = dict()
        for key in sample_batched.keys():
            if 'query' in key:
                batch_input[key] = sample_batched[key].float().cuda()
            else:
                batch_input[key] = sample_batched[key].long().cuda()
        query_embeddings = model.generate_query_emb(batch_input)
        if args.train_form == 'link':
            top_10_precision, top_5_precision = get_precision_metric(batch_input['s'].cpu(), query_embeddings.detach(), all_node_embeddings, model, truth, all_nodes, all_nei_embeddings)
        if args.train_form == 'path':
            top_10_precision, top_5_precision = get_precision_metric_p(batch_input['s'].cpu(), batch_input['t'].cpu(), query_embeddings.detach(), all_node_embeddings, model, truth, all_nodes, all_nei_embeddings)
        Top_10_list.append(top_10_precision)
        Top_5_list.append(top_5_precision)
        '''
        sub_mrr_list, sub_h1_list, sub_h3_list, sub_h10_list = get_rank_metric(batch_input['s'].cpu(), batch_input['t'].cpu(), query_embeddings.detach(), all_node_embeddings, model, train_graph)
        MRR_list += sub_mrr_list
        H1_list += sub_h1_list
        H3_list += sub_h3_list
        H10_list += sub_h10_list
        '''
    print("Top_10 precision: " + str(np.mean(Top_10_list)))
    print("Top_5 precision: " + str(np.mean(Top_5_list)))
    #print("Test MRR: "+str(np.mean(MRR_list)))
    #print("Test H1: "+str(np.mean(H1_list)))
    #print("Test H3: "+str(np.mean(H3_list)))
    #print("Test H10: "+str(np.mean(H10_list)))
    return np.mean(Top_10_list), np.mean(Top_5_list)



def retrival_task_eval(model, raw_graph, args, device, tokenizer, all_query_embs, train_graph):
    print("start retrival task eval...")
    cite_dict_id, id_2_title_abs_test = pickle.load(open("test_set.pkl","rb"))
    Node_dataset = NodeDataset(cite_dict_id, id_2_title_abs_test, args, device, tokenizer)

    print("generate embeddings for all nodes")
    loader = DataLoader(Node_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    all_node_embeddings = []
    all_nodes = []
    for i_batch, sample_batched in tqdm(enumerate(loader), disable=False, total=len(loader)):
        batch_input = dict()
        for key in sample_batched.keys():
            batch_input[key] = sample_batched[key].long().cuda()      
        subnode_embeddings = model.generate_candidate_emb(batch_input).detach()
        all_node_embeddings.append(subnode_embeddings)
        all_nodes.append(batch_input['target'].detach().cpu())
    all_node_embeddings = torch.cat(all_node_embeddings, 0)
    all_nodes = torch.cat(all_nodes, 0).squeeze(-1)
    print(all_node_embeddings.size())
    print(all_node_embeddings[0][:10])

    # get test query embeddings
    
    try:
        test_query_embs = pickle.load(open('test_query_embeddings_'+args.pretrained_model.replace('/','_')+'_'+args.train_form+'.pkl', 'rb'))
        print('---------------------------')
    except:
        # define pre-trained LLM for query embedding
        pretrained_model_name = args.pretrained_model
        config = AutoConfig.from_pretrained(pretrained_model_name)
        hidden_size = config.hidden_size
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        LLM_model = AutoModel.from_pretrained(pretrained_model_name).cuda()
        LLM_model.eval()

        test_query_embs = dict()
        tmp_list = cite_dict_id.keys()
        for key in tqdm(tmp_list):
            prompt = get_prompt(key, id_2_title_abs_test)

            encoded_input = tokenizer([prompt], padding = True, truncation=True, max_length=512 , return_tensors='pt')
            with torch.no_grad():
                output = LLM_model(**encoded_input.to('cuda'), output_hidden_states=True).hidden_states[-1]
                sentence_embedding = output[:, 0, :]

            test_query_embs[key] = sentence_embedding[0].cpu()
        pickle.dump(test_query_embs, open('test_query_embeddings_'+args.pretrained_model.replace('/','_')+'_'+args.train_form+'.pkl', 'wb'))
    
    Test_dataset = TestEdgeDataset(cite_dict_id, id_2_title_abs_test, args, device, tokenizer, test_query_embs)
    truth = cite_dict_id
    
    print(len(truth[list(truth.keys())[0]]))
    print("test set eval")
    MRR_list, H1_list, H3_list, H10_list = [], [], [], []
    Top_10_list, Top_5_list, Top_1_list = [], [], []
    all_querys, all_pred, all_truth = [], [], []
    loader = DataLoader(Test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    for i_batch, sample_batched in tqdm(enumerate(loader), disable=False, total=len(loader)):
        batch_input = dict()
        for key in sample_batched.keys():
            if 'query' in key:
                batch_input[key] = sample_batched[key].float().cuda()
            else:
                batch_input[key] = sample_batched[key].long().cuda()
        query_embeddings = model.generate_query_emb(batch_input)
        top_10_precision, top_5_precision, top_1_precision, querys, pred, sub_truth = get_precision_metric_sample(batch_input['s'].cpu(), query_embeddings.detach(), all_node_embeddings, model, truth, all_nodes)
        all_querys += querys
        all_pred += pred
        all_truth += sub_truth
        Top_10_list.append(top_10_precision)
        Top_5_list.append(top_5_precision)
        Top_1_list.append(top_1_precision)
        
        #sub_mrr_list, sub_h1_list, sub_h3_list, sub_h10_list = get_rank_metric(batch_input['s'].cpu(), batch_input['t'].cpu(), query_embeddings.detach(), all_node_embeddings, model, train_graph)
        #MRR_list += sub_mrr_list
        #H1_list += sub_h1_list
        #H3_list += sub_h3_list
        #H10_list += sub_h10_list
        
        
    print("Top_10 precision: " + str(np.mean(Top_10_list)))
    print("Top_5 precision: " + str(np.mean(Top_5_list)))
    print("Top_1 precision: " + str(np.mean(Top_1_list)))
    #print("Test MRR: "+str(np.mean(MRR_list)))
    #print("Test H1: "+str(np.mean(H1_list)))
    #print("Test H3: "+str(np.mean(H3_list)))
    #print("Test H10: "+str(np.mean(H10_list)))
    print([len(all_querys), len(all_pred), len(all_truth), len(all_pred[10])])
    #pickle.dump([all_querys, all_pred, all_truth], open('retrieval_result_bert.pkl', 'wb'))
    return np.mean(Top_10_list), np.mean(Top_5_list)

def main(args):
    setup_seed(1)

    # define model
    
    if args.model == 'Bert': # use this
        config = AutoConfig.from_pretrained(args.pretrained_model, output_hidden_states=True)
        config.heter_embed_size = args.heter_embed_size
        args.hidden_size = config.hidden_size
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        print("bert initialized")
        model = Bert_v4(config, args).cuda() #Bert_v4(config, args).cuda()
    print("model constructed")

    # read data
    raw_graph_data = pickle.load(open("subgraph.pickle","rb"))
    test_data = pickle.load(open('LP_node_size_10000_pos_2000.pickle', 'rb'))
    #raw_graph_translated, test_data_translated = translate_graph(raw_graph_data, test_data)
    
    train_dataset = TrainDataset(raw_graph_data, test_data, args, device, tokenizer)
    args.node_num = len(train_dataset.train_graph.nodes())
    print("data read")

    # get all query embeddings
    
    try:
        all_query_embs = pickle.load(open('llm_embeddings_bert_'+args.pretrained_model.replace('/','_')+'_'+args.train_form+'.pkl', 'rb'))
        print(all_query_embs[list(all_query_embs.keys())[0]])
        print('---------------------------')
    except:
        # define pre-trained LLM for query embedding
        pretrained_model_name = args.pretrained_model
        config = AutoConfig.from_pretrained(pretrained_model_name)
        hidden_size = config.hidden_size
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        LLM_model = AutoModel.from_pretrained(pretrained_model_name).cuda()
        LLM_model.eval()

        all_query_embs = dict()
        tmp_list = list(train_dataset.train_graph.nodes())
        tmp_dict = train_dataset.id_2_title_abs
        for key in tqdm(tmp_list):
            prompt = get_prompt(key, tmp_dict)
            encoded_input = tokenizer([prompt], padding = True, truncation=True, max_length=512 , return_tensors='pt')
            with torch.no_grad():
                output = LLM_model(**encoded_input.to('cuda'), output_hidden_states=True).hidden_states[-1]
                sentence_embedding = output[:, 0, :]

            all_query_embs[key] = sentence_embedding[0].cpu()
        pickle.dump(all_query_embs, open('llm_embeddings_bert_'+args.pretrained_model.replace('/','_')+'_'+args.train_form+'.pkl', 'wb'))
    print(all_query_embs[list(all_query_embs)[0]])
    train_dataset.init_query_embeddings(all_query_embs)
    
    # define optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
    scaler = GradScaler()
    
    # start training
    best_result = [0,0]
    print("start training")
    model.eval()
    with torch.no_grad():
        retrival_task_eval(model, raw_graph_data, args, device, tokenizer, train_dataset.query_embeddings, train_dataset.train_graph)
    for j in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
        for i_batch, sample_batched in tqdm(enumerate(loader), disable=False, total=len(loader)):
            batch_input = dict()
            for key in sample_batched.keys():
                if 'query' in key:
                    batch_input[key] = sample_batched[key].float().cuda()
                else:
                    batch_input[key] = sample_batched[key].long().cuda()
            
            if args.scale:
                with autocast():
                    loss = model(batch_input)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model(batch_input)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss = round((loss.detach().clone()).cpu().item(), 4)
            epoch_loss += loss / len(loader)
       
        print(str(j) + "th epoch mean loss: " + str(epoch_loss))
        print(' Best result 10: ' + str(best_result[0]))
        print(' Best result 5: ' + str(best_result[1]))
        if j % 5 == 0:
            model.eval()
            with torch.no_grad():
                P_10, P_5 = retrival_task_eval(model, raw_graph_data, args, device, tokenizer, train_dataset.query_embeddings, train_dataset.train_graph)
                if P_10 >= best_result[0] and P_5 >= best_result[1]:
                    best_result = [P_10, P_5]
    #torch.save(model,'save_best.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--max_length", type=int, default=48)
    parser.add_argument("--heter_embed_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32) # 12
    parser.add_argument("--neigh_num", type=int, default=4)
    parser.add_argument("--mask_ratio", type=float, default=0.2)
    parser.add_argument("--interact_form", type=str, default='add') # no_graph
    parser.add_argument("--score_func", type=str, default='TransE') # TransE
    parser.add_argument("--loss_func", type=str, default='Cross') # Margin
    parser.add_argument("--num_encode_layer", type=int, default=4)
    parser.add_argument("--num_bert_layer", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=51)
    parser.add_argument("--max_grad_norm", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=1e-3) # 1e-3
    parser.add_argument("--adam_epsilon", type=float, default=1e-8) # 1e-8
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--scale", type=bool, default=True) # Use small batch_size for False
    parser.add_argument("--model", type=str, default='Bert')
    parser.add_argument("--pretrained_model", type=str, default='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract') # ../bert-base-uncased # ../pmid_bert # lmsys/vicuna-7b-v1.5
    parser.add_argument("--train_form", type=str, default='QA') # link / path / QA
    args = parser.parse_args()

    device = torch.device("cuda:0")
    print("device:", device)
    main(args)


    '''
    lp_v4
    Best result 10: 0.799032738095238
    Best result 5: 0.760639880952381
    path_v4
    Top_10 precision: 0.04140625
    Top_5 precision: 0.017578125


    path_v3
    Best result 10: 0.015625
    Best result 5: 0.0078125
    '''