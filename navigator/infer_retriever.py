import os
import yaml
import torch
import json
import pickle
import random
import argparse
import numpy as np
from numpy.linalg import norm
from torch import nn
from tqdm import tqdm
import networkx as nx
import torch.nn.functional as F
#from torch_geometric.data import Data
from torch.utils.data import DataLoader 
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from transformers import AutoTokenizer, AutoConfig, AdamW, AutoModelForCausalLM, AutoModel, LlamaTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from Dataset import TrainDataset, NodeDataset, TestEdgeDataset, NodeDataset_QA, TestEdgeDataset_QA
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training
)

from Model import retriever

def get_prompt(key, id_2_text_dict):
    title, abstract = id_2_text_dict[key]
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


def get_result(s_ind, query_embs, all_node_embeddings, model, truth, all_nodes, id_2_raw_id_dict, raw_graph_data):
    # get score of candidate samples
    R_list, RR_list, H1_list, H3_list, H10_list = [], [], [], [], []
    top_10_result, top_5_result = [], []
    all_nodes = np.array(all_nodes)
    query_ind_list = []
    top10_list, top30_list = [], []
    candidate_list = []
    for i in range(query_embs.size()[0]):
        query_ind = s_ind[i]
        s_embeddings = query_embs[i]
        tmp_scores = model.Cos(s_embeddings, all_node_embeddings).flatten()
        sorted_scores, idxs = torch.sort(tmp_scores, descending = True)
        top_30 = all_nodes[[int(k) for k in idxs[:30]]]
        top_10 = all_nodes[[int(k) for k in idxs[:10]]]
        top_5 = all_nodes[[int(k) for k in idxs[:5]]]

        top30_list.append([id_2_raw_id_dict[i] for i in top_30])

        # valid candidates
        if truth != None:
            candidates = list(truth[int(s_ind[i])])
            top_10_hits = len(set(top_10) & set(candidates))
            top_5_hits = len(set(top_5) & set(candidates))
            top_10_result.append(float(top_10_hits)/min(10.0, len(candidates)))
            top_5_result.append(float(top_5_hits)/min(5.0, len(candidates)))

            query_ind_list.append(id_2_raw_id_dict[int(query_ind)])
            top10_list.append([id_2_raw_id_dict[i] for i in top_10])
            candidate_list.append([id_2_raw_id_dict[i] for i in candidates])
        else:
            print(query_ind)
            print(len(all_node_embeddings))
            query_ind_list.append(id_2_raw_id_dict[int(query_ind)])
            top10_list.append([id_2_raw_id_dict[i] for i in top_10])
            candidate_list.append([id_2_raw_id_dict[i] for i in top_10])

            for i in top_10:
                raw_id = id_2_raw_id_dict[i]
                print(raw_graph_data.nodes[raw_id]['Title'])
            print('-------------------------')
    
    return top30_list


def get_diversity_metric(tokenizer, top_K, id_2_raw_id_dict, raw_graph_data, LLM_model): # cosin sim
    title_list = []
    for i in top_K:
        raw_id = id_2_raw_id_dict[i]
        title_list.append(raw_graph_data.nodes()[raw_id]['Title'])
    
    encoded_input = tokenizer(title_list, padding = True, truncation=True, max_length=512 , return_tensors='pt')
    with torch.no_grad():
        output = LLM_model(**encoded_input.to('cuda'), output_hidden_states=True).hidden_states[-1]
        sentence_embedding = output[:, 0, :]
    
    all_sim = []
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    mean_embedding = torch.mean(sentence_embedding, dim = 0, keepdim = True)
    diversity_list = cos(mean_embedding.repeat(sentence_embedding.size(0), 1), sentence_embedding)
    diversity_rank = torch.argsort(diversity_list, 0, descending=True).cpu()
    all_rank = diversity_rank + torch.range(0, len(diversity_rank)-1, 1)
    selected_paper_index = torch.argsort(all_rank, 0, descending=False)
    selected_paper = []
    for index in selected_paper_index[:50]:
        selected_paper.append(top_K[index])
    
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    for j in selected_paper_index[:50]:
        for k in selected_paper_index[:50]:
            if k != j:
                sim = cos(sentence_embedding[k], sentence_embedding[j])
                all_sim.append(sim)
    all_sim = torch.stack(all_sim)
    return selected_paper, torch.mean(all_sim).detach().cpu()

def get_citenum_metric(gt_paper_list, predict_paper_list, id_2_raw_id, citenum_dict):
    gt_citenum_list, predict_citenum_list = [], []

    for paper in gt_paper_list:
        gt_citenum_list.append(int(citenum_dict[id_2_raw_id[paper]]))
    
    for paper in predict_paper_list:
        predict_citenum_list.append(int(citenum_dict[id_2_raw_id[paper]]))
    
    return gt_citenum_list, predict_citenum_list


def get_precision_metric(s_ind, query_embs, all_node_embeddings, model, truth, all_nodes, id_2_raw_id_dict, raw_graph_data, metric_tokenizer, metric_model, citenum_dict):
    
    # get score of candidate samples
    R_list, RR_list, H1_list, H3_list, H10_list = [], [], [], [], []
    top_50_result, top_10_result, top_5_result = [], [], []
    all_nodes = np.array(all_nodes)
    query_ind_list = []
    top10_list = []
    candidate_list = []

    diversity_metric_list = []
    citenum_metric_list = []
    for i in range(query_embs.size()[0]):
        query_ind = s_ind[i]
        s_embeddings = query_embs[i]
        tmp_scores = model.Cos(s_embeddings, all_node_embeddings).flatten()
        sorted_scores, idxs = torch.sort(tmp_scores, descending = True)
        top_100 = all_nodes[[int(k) for k in idxs[:100]]]
        top_50 = all_nodes[[int(k) for k in idxs[:50]]]
        top_10 = all_nodes[[int(k) for k in idxs[:10]]]
        top_5 = all_nodes[[int(k) for k in idxs[:5]]]

        candidates = list(truth[int(s_ind[i])])
        top_50, diversity = get_diversity_metric(metric_tokenizer, top_100, id_2_raw_id_dict, raw_graph_data, metric_model)
        true_citenum_list, predict_citenum_list = get_citenum_metric(candidates, all_nodes[[int(k) for k in idxs[:len(candidates)]]], id_2_raw_id_dict, citenum_dict)
        diversity_metric_list.append(diversity)
        citenum_metric_list.append([true_citenum_list, predict_citenum_list])

        # valid candidates
        if truth != None:
            top_50_hits = len(set(top_50) & set(candidates))
            top_10_hits = len(set(top_10) & set(candidates))
            top_5_hits = len(set(top_5) & set(candidates))
            
            top_50_result.append(float(top_50_hits)/min(50.0, len(candidates)))
            top_10_result.append(float(top_10_hits)/min(10.0, len(candidates)))
            top_5_result.append(float(top_5_hits)/min(5.0, len(candidates)))

            query_ind_list.append(id_2_raw_id_dict[int(query_ind)])
            top10_list.append([id_2_raw_id_dict[i] for i in top_10])
            candidate_list.append([id_2_raw_id_dict[i] for i in candidates])
        else:
            query_ind_list.append(id_2_raw_id_dict[int(query_ind)])
            top10_list.append([id_2_raw_id_dict[i] for i in top_10])
            candidate_list.append([id_2_raw_id_dict[i] for i in top_10])
    
    return top_50_result, top_10_result, diversity_metric_list, query_ind_list, top10_list, candidate_list, citenum_metric_list


def retrival_for_all_test_nodes(model, graph_data, args, device, tokenizer, raw_graph_data, id_2_raw_id_dict):
    print("start retrival task eval...")

    cite_dict_id = graph_data[1]
    raw_graph, id_2_title_abs = graph_data[0], graph_data[2]

    print('generate and save candidate embeddings')
    
    all_query_embs = pickle.load(open('../llm_embeddings_'+args.pretrained_model.replace('/','_')+'_'+args.dataset_name+'.pkl', 'rb'))
    Node_dataset = NodeDataset_QA(raw_graph, id_2_title_abs, args, device, tokenizer)
    Node_dataset.init_bert_embeddings(all_query_embs)

    loader = DataLoader(Node_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    all_node_embeddings = []
    all_nodes = []
    for i_batch, sample_batched in tqdm(enumerate(loader), disable=False, total=len(loader)):
        batch_input = dict()
        for key in sample_batched.keys():
            batch_input[key] = sample_batched[key].float().cuda()      
        subnode_embeddings = model.generate_candidate_emb(batch_input).detach()
        all_node_embeddings.append(subnode_embeddings)
        all_nodes.append(batch_input['target'].detach().cpu())
    all_node_embeddings = torch.cat(all_node_embeddings, 0)
    all_nodes = torch.cat(all_nodes, 0).squeeze(-1)
    print(all_node_embeddings.size())
    print(all_node_embeddings[0][:10])
    
    '''
    try:
        all_nodes, all_node_embeddings = pickle.load(open('all_candidate_embeddings_'+args.pretrained_model.replace('/','_')+'_'+args.train_form+''+args.dataset_name+'.pkl', 'rb'))
        print('---------------------------')
    except:
        print('generate and save candidate embeddings')
        all_query_embs = pickle.load(open('llm_embeddings_'+args.pretrained_model.replace('/','_')+'_'+args.dataset_name+'.pkl', 'rb'))
        Node_dataset = NodeDataset_QA(raw_graph, id_2_title_abs, args, device, tokenizer)
        Node_dataset.init_bert_embeddings(all_query_embs)

        loader = DataLoader(Node_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
        all_node_embeddings = []
        all_nodes = []
        for i_batch, sample_batched in tqdm(enumerate(loader), disable=False, total=len(loader)):
            batch_input = dict()
            for key in sample_batched.keys():
                batch_input[key] = sample_batched[key].float.cuda()      
            subnode_embeddings = model.generate_candidate_emb(batch_input).detach()
            all_node_embeddings.append(subnode_embeddings)
            all_nodes.append(batch_input['target'].detach().cpu())
        all_node_embeddings = torch.cat(all_node_embeddings, 0)
        all_nodes = torch.cat(all_nodes, 0).squeeze(-1)
        print(all_node_embeddings.size())
        print(all_node_embeddings[0][:10])
        pickle.dump([all_nodes, all_node_embeddings], open('all_candidate_embeddings_'+args.pretrained_model.replace('/','_')+'_'+args.train_form+''+args.dataset_name+'.pkl', 'wb'))
    '''

    print('generate query embeddings...')
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
        prompt = get_prompt(key, id_2_title_abs)

        encoded_input = tokenizer([prompt], padding = True, truncation=True, max_length=512 , return_tensors='pt')
        with torch.no_grad():
            output = LLM_model(**encoded_input.to('cuda'), output_hidden_states=True).hidden_states[-1]
            sentence_embedding = output[:, 0, :]

        test_query_embs[key] = sentence_embedding[0].cpu()

    '''
    # get test query embeddings
    try:
        test_query_embs = pickle.load(open('test_query_embeddings_bert_'+args.pretrained_model.replace('/','_')+'_'+args.train_form+''+args.dataset_name+'.pkl', 'rb'))
        print('---------------------------')
    except:
        print('generate query embeddings...')
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
            prompt = get_prompt(key, id_2_title_abs)

            encoded_input = tokenizer([prompt], padding = True, truncation=True, max_length=512 , return_tensors='pt')
            with torch.no_grad():
                output = LLM_model(**encoded_input.to('cuda'), output_hidden_states=True).hidden_states[-1]
                sentence_embedding = output[:, 0, :]

            test_query_embs[key] = sentence_embedding[0].cpu()
        pickle.dump(test_query_embs, open('test_query_embeddings_bert_'+args.pretrained_model.replace('/','_')+'_'+args.train_form+''+args.dataset_name+'.pkl', 'wb'))
    '''
    
    # init metric model
    pretrained_model_name = args.pretrained_model
    config = AutoConfig.from_pretrained(pretrained_model_name)
    hidden_size = config.hidden_size
    metric_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    metric_model = AutoModel.from_pretrained(pretrained_model_name).cuda()
    metric_model.eval()

    # citenum file
    citenum_dict = json.load(open('/home/jz875/project/Citation_Graph/LLM_3_37/navigator/citation_count.json', 'r'))
    
    Test_dataset = TestEdgeDataset(cite_dict_id, id_2_title_abs, args, device, tokenizer, test_query_embs)
    truth = cite_dict_id
    print(len(cite_dict_id.keys()))
    print("test set eval")
    MRR_list, H1_list, H3_list, H10_list = [], [], [], []
    Top_10_list, Top_5_list = [], []
    all_querys, all_pred, all_truth = [], [], []
    Diversity_metric = []
    Citenum_metric = []
    loader = DataLoader(Test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    for i_batch, sample_batched in tqdm(enumerate(loader), disable=False, total=len(loader)):
        batch_input = dict()
        for key in sample_batched.keys():
            if 'query' in key:
                batch_input[key] = sample_batched[key].float().cuda()
            else:
                batch_input[key] = sample_batched[key].long().cuda()
        query_embeddings = model.generate_query_emb(batch_input)
        top_10_precision, top_5_precision, diversity_metric, querys, pred, sub_truth, citenum_metric = get_precision_metric(batch_input['s'].cpu(), query_embeddings.detach(), all_node_embeddings, model, truth, all_nodes, id_2_raw_id_dict, raw_graph_data, metric_tokenizer, metric_model, citenum_dict)
        all_querys += querys
        all_pred += pred
        all_truth += sub_truth
        Top_10_list += top_10_precision
        Top_5_list += top_5_precision
        Diversity_metric += diversity_metric
        Citenum_metric += citenum_metric
        '''
        sub_mrr_list, sub_h1_list, sub_h3_list, sub_h10_list = get_rank_metric(batch_input['s'].cpu(), batch_input['t'].cpu(), query_embeddings.detach(), all_node_embeddings, model, train_graph)
        MRR_list += sub_mrr_list
        H1_list += sub_h1_list
        H3_list += sub_h3_list
        H10_list += sub_h10_list
        '''
    print("Top_10 precision: " + str(np.mean(Top_10_list)))
    print("Top_5 precision: " + str(np.mean(Top_5_list)))
    print("diversity: " + str(np.mean(Diversity_metric)))
    #print("Test MRR: "+str(np.mean(MRR_list)))
    #print("Test H1: "+str(np.mean(H1_list)))
    #print("Test H3: "+str(np.mean(H3_list)))
    #print("Test H10: "+str(np.mean(H10_list)))

    true_count_list = []
    true_citenum_list = []
    pred_citenum_list = []
    for item in Citenum_metric:
        true_count_list.append(len(item[0]))

        true_citenum_list += item[0]
        pred_citenum_list += item[1]
    
    true_hist, _ = np.histogram(true_citenum_list, bins=100)
    pred_hist, _ = np.histogram(pred_citenum_list, bins=100)
    
    print(true_hist)
    print(pred_hist)
    print([np.mean(true_count_list), np.mean(true_citenum_list), np.mean(pred_citenum_list), np.dot(np.log(true_hist+1), np.log(pred_hist+1)) / (norm(np.log(true_hist+1)) * norm(np.log(pred_hist+1)))])

    print([len(all_querys), len(all_pred), len(all_truth), len(all_pred[10])])


def overlap_test(model, graph_data, args, device, tokenizer, all_query_embs, id_2_raw_id_dict, raw_graph_data):
    print("generate embeddings for all candidate nodes")

    cite_dict_id = graph_data[1]
    
    raw_graph, id_2_title_abs = graph_data[0], graph_data[2]

    all_query_embs = pickle.load(open('../llm_embeddings_'+args.pretrained_model.replace('/','_')+'_'+args.dataset_name+'.pkl', 'rb'))
    Node_dataset = NodeDataset_QA(raw_graph, id_2_title_abs, args, device, tokenizer)
    Node_dataset.init_bert_embeddings(all_query_embs)

    loader = DataLoader(Node_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    all_node_embeddings = []
    all_nodes = []
    for i_batch, sample_batched in tqdm(enumerate(loader), disable=False, total=len(loader)):
        batch_input = dict()
        for key in sample_batched.keys():
            batch_input[key] = sample_batched[key].float().cuda()      
        subnode_embeddings = model.generate_candidate_emb(batch_input).detach()
        all_node_embeddings.append(subnode_embeddings)
        all_nodes.append(batch_input['target'].detach().cpu())
    all_node_embeddings = torch.cat(all_node_embeddings, 0)
    all_nodes = torch.cat(all_nodes, 0).squeeze(-1)
    print(all_node_embeddings.size())
    print(all_node_embeddings[0][:10])

    # get test query embeddings
    print("generate embeddings for test paper")
    test_query_embs = {}
    print('---------------------------')
    pretrained_model_name = args.pretrained_model
    config = AutoConfig.from_pretrained(pretrained_model_name)
    hidden_size = config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    LLM_model = AutoModel.from_pretrained(pretrained_model_name).cuda()
    LLM_model.eval()

    test_pairs = []
    test_pairs.append(['LitFM: A Retrieval Augmented Structure-aware Foundation Model For Citation Graphs', 'Learning to Generate Novel Scientific Directions with Contextualized Literature-based Discovery'])
    test_pairs.append(['DTGB: A Comprehensive Benchmark for Dynamic Text-Attributed Graphs', 'Temporal Graph Benchmark for Machine Learning on Temporal Graphs'])
    test_pairs.append(['Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View', 'Graph Neural Networks for Social Recommendation'])
    test_pairs.append(['ImageNet Classification with Deep Convolutional Neural Networks', 'Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network'])
    test_pairs.append(['Temporal Knowledge Graph Reasoning with Historical Contrastive Learning', 'Multi-hop temporal knowledge graph reasoning with multi-agent reinforcement learning'])
    
    all_results = []
    for pair in test_pairs:
        encoded_input = tokenizer(pair, padding = True, truncation=True, max_length=512 , return_tensors='pt')
        with torch.no_grad():
            output = LLM_model(**encoded_input.to('cuda'), output_hidden_states=True).hidden_states[-1]
            sentence_embedding = output[:, 0, :]

        test_query_embs[0] = sentence_embedding[0].cpu()
        test_query_embs[1] = sentence_embedding[1].cpu()
        
        # init test dataset
        Test_dataset = TestEdgeDataset_QA(args, device, tokenizer, test_query_embs)
        print("test set eval")

        MRR_list, H1_list, H3_list, H10_list = [], [], [], []
        Top_10_list, Top_5_list = [], []
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
            all_results.append(get_result(batch_input['s'].cpu(), query_embeddings.detach(), all_node_embeddings, model, None, all_nodes, id_2_raw_id_dict, raw_graph_data))
        print([len(all_results), len(all_results[0])])
        print(all_results)
        ratio = 0
        for i in all_results:
            ratio += float(len(set(i[0]) & set(i[1]))) / float(len(i[0]))
            print(ratio)
        print(ratio/len(all_results))

def retrival_for_one_paper(model, graph_data, args, device, tokenizer, all_query_embs, id_2_raw_id_dict, raw_graph_data):
    print("generate embeddings for all candidate nodes")
    raw_graph, id_2_title_abs = graph_data[0], graph_data[2]
    Node_dataset = NodeDataset_QA(raw_graph, id_2_title_abs, args, device, tokenizer)

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
    print("generate embeddings for test paper")
    test_query_embs = {}
    print('---------------------------')
    pretrained_model_name = args.pretrained_model
    config = AutoConfig.from_pretrained(pretrained_model_name)
    hidden_size = config.hidden_size
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    LLM_model = AutoModel.from_pretrained(pretrained_model_name).cuda()
    LLM_model.eval()
    #prompt_title = "LitFM: A Retrieval Augmented Structure-aware Foundation Model For Citation Graphs"
    #prompt_abs = "With the advent of large language models (LLMs), managing scientific literature via LLMs has become a promising direction of research. However, existing approaches often overlook the rich structural and semantic relevance among scientific literature, limiting their ability to discern the relationships between pieces of scientific knowledge, and suffer from various types of hallucinations. These methods also focus narrowly on individual downstream tasks, limiting their applicability across use cases. Here we propose LitFM, the first literature foundation model designed for a wide variety of practical downstream tasks on domain-specific literature, with a focus on citation information. At its core, LitFM contains a novel graph retriever to integrate graph structure by navigating citation graphs and extracting relevant literature, thereby enhancing model reliability. LitFM also leverages a knowledge-infused LLM, fine-tuned through a well-developed instruction paradigm. It enables LitFM to extract domain-specific knowledge from literature and reason relationships among them. By integrating citation graphs during both training and inference, LitFM can generalize to unseen papers and accurately assess their relevance within existing literature. Additionally, we introduce new large-scale literature citation benchmark datasets on three academic fields, featuring sentence-level citation information and local context. Extensive experiments validate the superiority of LitFM, achieving 28.1% improvement on retrieval task in precision, and an average improvement of 7.52% over state-of-the-art across six downstream literature-related tasks"
    
    prompt_title = "DTGB: A Comprehensive Benchmark for Dynamic Text-Attributed Graphs"
    prompt_abs = "Dynamic text-attributed graphs (DyTAGs) are prevalent in various real-world scenarios, where each node and edge are associated with text descriptions, and both the graph structure and text descriptions evolve over time. Despite their broad applicability, there is a notable scarcity of benchmark datasets tailored to DyTAGs, which hinders the potential advancement in many research fields. To address this gap, we introduce Dynamic Text-attributed Graph Benchmark (DTGB), a collection of large-scale, time-evolving graphs from diverse domains, with nodes and edges enriched by dynamically changing text attributes and categories. To facilitate the use of DTGB, we design standardized evaluation procedures based on four real-world use cases: future link prediction, destination node retrieval, edge classification, and textual relation generation. These tasks require models to understand both dynamic graph structures and natural language, highlighting the unique challenges posed by DyTAGs. Moreover, we conduct extensive benchmark experiments on DTGB, evaluating 7 popular dynamic graph learning algorithms and their variants of adapting to text attributes with LLM embeddings, along with 6 powerful large language models (LLMs). Our results show the limitations of existing models in handling DyTAGs. Our analysis also demonstrates the utility of DTGB in investigating the incorporation of structural and textual dynamics. The proposed DTGB fosters research on DyTAGs and their broad applications. It offers a comprehensive benchmark for evaluating and advancing models to handle the interplay between dynamic graph structures and natural language."
    control_word = "Benchmark datasets for dynamic graphs"

    for control_word in ['Techniques for graph representation learning', 'Recent advances in text-attributed graph mining tasks', 'foundation models for citation graphs', 'retrieval augmentation techniques for structure-aware foundation models', 'graph neural networks for citation graph analysis']:
        prompt = control_word #prompt_title + prompt_abs + control_word
        encoded_input = tokenizer([prompt], padding = True, truncation=True, max_length=512 , return_tensors='pt')
        with torch.no_grad():
            output = LLM_model(**encoded_input.to('cuda'), output_hidden_states=True).hidden_states[-1]
            sentence_embedding = output[:, 0, :]

        test_query_embs[0] = sentence_embedding[0].cpu()
        
        # init test dataset
        Test_dataset = TestEdgeDataset_QA(args, device, tokenizer, test_query_embs)
        print("test set eval")

        MRR_list, H1_list, H3_list, H10_list = [], [], [], []
        Top_10_list, Top_5_list = [], []
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
            get_result(batch_input['s'].cpu(), query_embeddings.detach(), all_node_embeddings, model, None, all_nodes, id_2_raw_id_dict, raw_graph_data)

def translate_graph(graph, raw_test_nodes):
    all_nodes = list(graph.nodes())
    raw_id_2_id_dict = {}
    id_2_raw_id_dict = {}

    num = 0
    for node in all_nodes:
        raw_id_2_id_dict[node] = num
        id_2_raw_id_dict[num] = node
        num += 1
    
    new_graph = nx.Graph()
    test_edges = []
    for edge in list(graph.edges()):
        h_id, t_id = raw_id_2_id_dict[edge[0]], raw_id_2_id_dict[edge[1]]
        if edge[0] in raw_test_nodes or edge[1] in raw_test_nodes:
            test_edges.append([h_id, t_id])
        else:
            new_graph.add_edge(h_id, t_id)
    
    return new_graph, raw_id_2_id_dict, id_2_raw_id_dict, test_edges


def main(args):
    setup_seed(1)
    args.dataset_name = 'CS'

    # define model
    if args.model == 'Bert': # use this
        config = AutoConfig.from_pretrained(args.pretrained_model, output_hidden_states=True)
        config.heter_embed_size = args.heter_embed_size
        args.hidden_size = config.hidden_size
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        print("bert initialized")
        model = retriever(config, args).cuda() #torch.load('save_best_1211'+args.pretrained_model+'_'+args.dataset_name+'.pt').cuda() #torch.load('save_best.pt').cuda() #Bert_v4(config, args).cuda() #Bert_v4(config, args).cuda()
    print("model constructed")

    if args.dataset_name == 'CS':
        raw_graph_data = nx.read_gexf("/home/jz875/project/Citation_Graph/cs_final_processed_with_attributes.gexf", node_type=None, relabel=False, version='1.2draft')
        raw_test_nodes = list(raw_graph_data.nodes())[:1000]
        graph_data, raw_id_2_id_dict, id_2_raw_id_dict, test_data = translate_graph(raw_graph_data, raw_test_nodes)
        print([len(graph_data.nodes()), len(graph_data.edges()), len(raw_test_nodes)])

    
    id_2_tile_abs = dict()
    for paper_id in raw_graph_data.nodes():
        title = raw_graph_data.nodes()[paper_id]['Title']
        abstract = raw_graph_data.nodes()[paper_id]['Abstract']
        id_2_tile_abs[raw_id_2_id_dict[paper_id]] = [title, abstract]


    id_pair_2_sentence = dict()
    for edge in list(raw_graph_data.edges()):
        sentence = raw_graph_data.edges()[edge]['Sentence']
        id_pair_2_sentence[(raw_id_2_id_dict[edge[0]], raw_id_2_id_dict[edge[1]])] = sentence

    test_GT = dict()
    test_node_ids = [raw_id_2_id_dict[i] for i in raw_test_nodes]
    for sample in test_data:
        h, t = sample
        if h in test_node_ids:
            if h not in test_GT:
                test_GT[h] = set()
            test_GT[h].add(t)
        

    
    graph_data = [graph_data, test_GT, id_2_tile_abs, id_pair_2_sentence]
    print("data read")
    
    model.eval()
    with torch.no_grad(): # test in citation dataset to verify the model
        overlap_test(model, graph_data, args, device, tokenizer, None, id_2_raw_id_dict, raw_graph_data)
        #retrival_for_one_paper(model, graph_data, args, device, tokenizer, None, id_2_raw_id_dict, raw_graph_data)
        #retrival_for_all_test_nodes(model, graph_data, args, device, tokenizer, raw_graph_data, id_2_raw_id_dict)
        #P_10, P_5 = retrival_task_eval(model, raw_graph_data, args, device, tokenizer, link_query_embs)
        #retrival_task_eval_qa(model, raw_graph_data, args, device, tokenizer, qa_query_embs)


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
    parser.add_argument("--pretrained_model", type=str, default='BAAI/bge-large-en-v1.5') # BAAI/bge-large-en-v1.5 # ../bert-base-uncased # ../pmid_bert # lmsys/vicuna-7b-v1.5
    parser.add_argument("--train_form", type=str, default='link') # link / path / QA
    args = parser.parse_args()

    device = torch.device("cuda:0")
    print("device:", device)
    main(args)