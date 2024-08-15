from torch.utils.data import Dataset
import networkx as nx
import numpy as np
import random
import torch

class TrainDataset(Dataset):
    def __init__(self, raw_graph, test_data, args, device, tokenizer=None):
        
        self.args = args
        self.device = device
        self.tokenizer = tokenizer

        self.test_edges = [(test_sample[0], test_sample[1]) for test_sample in test_data]
        self.id_2_title_abs = raw_graph[2]
        self.id_2_sent_cont = raw_graph[3]
        
        # filter the test samples from train graph
        self.train_graph = nx.Graph()
        for edge in raw_graph[0].edges():
            if edge in self.test_edges:
                continue
            self.train_graph.add_edge(edge[0], edge[1])
        
        print(self.id_2_sent_cont[list(self.id_2_sent_cont.keys())[0]][0])
        
        self.train_edges = list(self.train_graph.edges())
        
        print("number of nodes: " + str(len(self.train_graph.nodes())))
        print("number of edges: " + str(len(self.train_graph.edges())))
        print("number of test samples: " + str(len(self.test_edges)))
    
    def init_query_embeddings(self, embs):
        self.query_embeddings = embs

    def __len__(self):
        return len(self.train_edges)
    
    def sample_neighbor(self, all_neighbor_list):
        if len(all_neighbor_list) >= self.args.neigh_num:
            return random.sample(all_neighbor_list, self.args.neigh_num)
        else:
            return all_neighbor_list+[-1]*self.args.neigh_num

    def __getitem__(self, idx):
        sampled_s_node, sampled_t_node = self.train_edges[idx]

        # query emb
        sentence_embedding = self.query_embeddings[sampled_s_node]

        # token of target nodes
        t_title, t_abs = self.id_2_title_abs[sampled_t_node]
        encoded_text = self.tokenizer.batch_encode_plus([t_title], max_length=self.args.max_length, padding='max_length', truncation=True)
        t_title_tokens, t_title_masks = encoded_text['input_ids'], encoded_text['attention_mask']
        encoded_text = self.tokenizer.batch_encode_plus([t_abs], max_length=self.args.max_length, padding='max_length', truncation=True)
        t_abs_tokens, t_abs_masks = encoded_text['input_ids'], encoded_text['attention_mask']

        # token of 1-hop neighbor of target nodes
        t_neighbors = self.sample_neighbor(list(nx.all_neighbors(self.train_graph, sampled_t_node)))
        t_neighbors_node_title_tokens, t_neighbors_node_title_masks = [], []
        t_neighbors_node_abs_tokens, t_neighbors_node_abs_masks = [], []

        for i in range(self.args.neigh_num):
            node = t_neighbors[i]
            if node == -1:
                node_title_text, node_abs_text = '', ''
            else:
                node_title_text, node_abs_text = self.id_2_title_abs[node]
            encoded_text = self.tokenizer.batch_encode_plus([node_title_text], max_length=self.args.max_length, padding='max_length', truncation=True)
            t_neighbors_node_title_tokens.append(encoded_text['input_ids'])
            t_neighbors_node_title_masks.append(encoded_text['attention_mask'])
            encoded_text = self.tokenizer.batch_encode_plus([node_abs_text], max_length=self.args.max_length, padding='max_length', truncation=True)
            t_neighbors_node_abs_tokens.append(encoded_text['input_ids'])
            t_neighbors_node_abs_masks.append(encoded_text['attention_mask'])
        
        sample = {
            "s": np.array([sampled_s_node]),
            "t": np.array([sampled_t_node]),
            "query_emb": np.array(sentence_embedding),
            "target_title_tokens": np.array(t_title_tokens),
            "target_title_masks": np.array(t_title_masks),
            "target_abs_tokens": np.array(t_abs_tokens),
            "target_abs_masks": np.array(t_abs_masks),
            "t_nei_n_title_tokens": np.array(t_neighbors_node_title_tokens),
            "t_nei_n_title_masks": np.array(t_neighbors_node_title_masks),
            "t_nei_n_abs_tokens": np.array(t_neighbors_node_abs_tokens),
            "t_nei_n_abs_masks": np.array(t_neighbors_node_abs_masks),
        }

        return sample

class TestEdgeDataset(Dataset):
    def __init__(self, cite_dict_id, id_2_title_abs, args, device, tokenizer=None, all_query_embs=None):
        self.args = args
        self.device = device
        self.tokenizer = tokenizer
        self.cite_dict_id = cite_dict_id
        self.query_embeddings = all_query_embs
        self.test_papers = list(self.cite_dict_id.keys())
    
    def __len__(self):
        return len(self.test_papers)

    def __getitem__(self, idx):
        sampled_s_node = self.test_papers[idx]

        # query emb
        sentence_embedding = self.query_embeddings[sampled_s_node]
        
        sample = {
            "s": np.array([sampled_s_node]),
            "query_emb": np.array(sentence_embedding)
        }
        
        return sample


class TestEdgeDataset_QA(Dataset):
    def __init__(self, args, device, tokenizer=None, all_query_embs=None):
        self.args = args
        self.device = device
        self.tokenizer = tokenizer
        self.query_embeddings = all_query_embs
        self.test_questions = list(all_query_embs.keys())
    
    def __len__(self):
        return len(self.test_questions)

    def __getitem__(self, idx):
        sampled_s_node = self.test_questions[idx]

        # query emb
        sentence_embedding = self.query_embeddings[sampled_s_node]
        
        sample = {
            "s": np.array([int(sampled_s_node)]),
            "query_emb": np.array(sentence_embedding)
        }
        
        return sample


class NodeDataset(Dataset):
    def __init__(self, cite_dict_id, id_2_title_abs, args, device, tokenizer=None):
        self.args = args
        self.device = device
        self.tokenizer = tokenizer
        self.id_2_title_abs = id_2_title_abs
        self.cite_dict_id = cite_dict_id
        self.all_nodes = list(self.id_2_title_abs.keys())

        self.test_graph = nx.Graph()
        for key in cite_dict_id.keys():
            nei_list = cite_dict_id[key]
            for i in nei_list:
                self.test_graph.add_edge(key, i)

    def __len__(self):
        return len(self.all_nodes)

    def sample_neighbor(self, all_neighbor_list):
        if len(all_neighbor_list) >= self.args.neigh_num:
            return random.sample(all_neighbor_list, self.args.neigh_num)
        else:
            return all_neighbor_list+[-1]*self.args.neigh_num
    
    def __getitem__(self, idx):
        sampled_node = self.all_nodes[idx]

        # text of sampled node
        node_title, node_abs = self.id_2_title_abs[sampled_node]
        if node_title == None:
            node_title = 'None'
        if node_abs == None:
            node_abs = 'None'
        encoded_text = self.tokenizer.batch_encode_plus([node_title], max_length=self.args.max_length, padding='max_length', truncation=True)
        node_title_tokens, node_title_masks = encoded_text['input_ids'], encoded_text['attention_mask']
        encoded_text = self.tokenizer.batch_encode_plus([node_abs], max_length=self.args.max_length, padding='max_length', truncation=True)
        node_abs_tokens, node_abs_masks = encoded_text['input_ids'], encoded_text['attention_mask']

        # token of 1-hop neighbor of target nodes
        t_neighbors = self.sample_neighbor(list(nx.all_neighbors(self.test_graph, sampled_node)))
        t_neighbors_node_title_tokens, t_neighbors_node_title_masks = [], []
        t_neighbors_node_abs_tokens, t_neighbors_node_abs_masks = [], []

        for i in range(self.args.neigh_num):
            node = t_neighbors[i]
            if node == -1:
                node_title_text, node_abs_text = '', ''
            else:
                node_title_text, node_abs_text = self.id_2_title_abs[node]
                if node_title_text == None:
                    node_title_text = 'None'
                if node_abs_text == None:
                    node_abs_text = 'None'
            encoded_text = self.tokenizer.batch_encode_plus([node_title_text], max_length=self.args.max_length, padding='max_length', truncation=True)
            t_neighbors_node_title_tokens.append(encoded_text['input_ids'])
            t_neighbors_node_title_masks.append(encoded_text['attention_mask'])
            encoded_text = self.tokenizer.batch_encode_plus([node_abs_text], max_length=self.args.max_length, padding='max_length', truncation=True)
            t_neighbors_node_abs_tokens.append(encoded_text['input_ids'])
            t_neighbors_node_abs_masks.append(encoded_text['attention_mask'])
        
        sample = {
            "target": np.array([sampled_node]),
            "target_title_tokens": np.array(node_title_tokens),
            "target_title_masks": np.array(node_title_masks),
            "target_abs_tokens": np.array(node_abs_tokens),
            "target_abs_masks": np.array(node_abs_masks),
            "t_nei_n_title_tokens": np.array(t_neighbors_node_title_tokens),
            "t_nei_n_title_masks": np.array(t_neighbors_node_title_masks),
            "t_nei_n_abs_tokens": np.array(t_neighbors_node_abs_tokens),
            "t_nei_n_abs_masks": np.array(t_neighbors_node_abs_masks),
        }
        
        return sample


class NodeDataset_QA(Dataset):
    def __init__(self, raw_graph, id_2_title_abs, args, device, tokenizer=None):
        self.args = args
        self.device = device
        self.tokenizer = tokenizer
        self.raw_graph = raw_graph
        self.id_2_title_abs = id_2_title_abs
        self.all_nodes = list(raw_graph.nodes())

    def __len__(self):
        return len(self.all_nodes)

    def sample_neighbor(self, all_neighbor_list):
        if len(all_neighbor_list) >= self.args.neigh_num:
            return random.sample(all_neighbor_list, self.args.neigh_num)
        else:
            return all_neighbor_list+[-1]*self.args.neigh_num
    
    def __getitem__(self, idx):
        sampled_node = self.all_nodes[idx]

        # text of sampled node
        node_title, node_abs = self.id_2_title_abs[sampled_node]
        if node_title == None:
            node_title = 'None'
        if node_abs == None:
            node_abs = 'None'
        encoded_text = self.tokenizer.batch_encode_plus([node_title], max_length=self.args.max_length, padding='max_length', truncation=True)
        node_title_tokens, node_title_masks = encoded_text['input_ids'], encoded_text['attention_mask']
        encoded_text = self.tokenizer.batch_encode_plus([node_abs], max_length=self.args.max_length, padding='max_length', truncation=True)
        node_abs_tokens, node_abs_masks = encoded_text['input_ids'], encoded_text['attention_mask']

        # token of 1-hop neighbor of target nodes
        t_neighbors = self.sample_neighbor(list(nx.all_neighbors(self.raw_graph, sampled_node)))
        t_neighbors_node_title_tokens, t_neighbors_node_title_masks = [], []
        t_neighbors_node_abs_tokens, t_neighbors_node_abs_masks = [], []

        for i in range(self.args.neigh_num):
            node = t_neighbors[i]
            if node == -1:
                node_title_text, node_abs_text = '', ''
            else:
                node_title_text, node_abs_text = self.id_2_title_abs[node]
                if node_title_text == None:
                    node_title_text = 'None'
                if node_abs_text == None:
                    node_abs_text = 'None'
            encoded_text = self.tokenizer.batch_encode_plus([node_title_text], max_length=self.args.max_length, padding='max_length', truncation=True)
            t_neighbors_node_title_tokens.append(encoded_text['input_ids'])
            t_neighbors_node_title_masks.append(encoded_text['attention_mask'])
            encoded_text = self.tokenizer.batch_encode_plus([node_abs_text], max_length=self.args.max_length, padding='max_length', truncation=True)
            t_neighbors_node_abs_tokens.append(encoded_text['input_ids'])
            t_neighbors_node_abs_masks.append(encoded_text['attention_mask'])
        
        sample = {
            "target": np.array([sampled_node]),
            "target_title_tokens": np.array(node_title_tokens),
            "target_title_masks": np.array(node_title_masks),
            "target_abs_tokens": np.array(node_abs_tokens),
            "target_abs_masks": np.array(node_abs_masks),
            "t_nei_n_title_tokens": np.array(t_neighbors_node_title_tokens),
            "t_nei_n_title_masks": np.array(t_neighbors_node_title_masks),
            "t_nei_n_abs_tokens": np.array(t_neighbors_node_abs_tokens),
            "t_nei_n_abs_masks": np.array(t_neighbors_node_abs_masks),
        }
        
        return sample
