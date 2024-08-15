# .mean(2)
# torch.ones -> torch.zeros
# all_node_embeddings = self.node_embedding[all_node_indexs]
# all_node_mask_inputs_title[:, None, None, :]
# delete all placeholder --->  and thus  s_nei_title_emb = s_nei_title_emb.reshape(batch_size, self.args.neigh_num, self.args.max_length, -1) changed
# add input_embedding_edge_text = input_embedding_edge_text / input_embedding_edge_text.norm(dim=-1, keepdim=True) for all embedidngs 

import torch
import numpy as np
from torch import nn
#from Encoder import NE_Encoder_new
import torch.nn.functional as F
from collections import OrderedDict
#from torch_geometric.nn import GATConv
#from Decoder import SeqDecoder, GraphDecoder

from transformers import AutoModel, BertLMHeadModel, AutoModelForCausalLM
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPreTrainedModel, BertModel
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType

class Bert(nn.Module):
    def __init__(self, config, args):
        super(Bert, self).__init__()
        self.args = args
        self.model = AutoModel.from_pretrained(args.pretrained_model)
        self.project = nn.Linear(768, args.hidden_size) #torch.nn.Sequential(nn.Linear(768, args.hidden_size), nn.ReLU(inplace=True), nn.Linear(args.hidden_size, args.hidden_size)) #torch.nn.Sequential( nn.Linear(4096, args.hidden_size), nn.ReLU(inplace=True), nn.Linear(args.hidden_size, 2*args.hidden_size))
        self.nei_aggregate = nn.Linear(2*args.hidden_size, args.hidden_size) #nn.Linear(2*args.hidden_size, 2*args.hidden_size)

    def cal_cl_loss(self, s_features, t_features, labels):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        logits = logit_scale * self.Cos(s_features, t_features)
        loss_i = F.cross_entropy(logits, labels)
        #loss_t = F.cross_entropy(logits.T, labels)
        ret_loss = loss_i
        return ret_loss
    
    def cal_cl_loss_EU(self, s_features, t_features, labels):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        logits = - logit_scale * self.TransE(s_features.unsqueeze(1), t_features.unsqueeze(0))
        loss_i = F.cross_entropy(logits, labels)
        #loss_t = F.cross_entropy(logits.T, labels)
        ret_loss = loss_i
        return ret_loss

    def Cos(self, h, t):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)

        score = h @ t.t()
        return score
    
    
    def TransE(self, h, t):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)

        score = h - t

        score = torch.norm(score, 2, -1)
        return score
    
    def forward(self, batched_data):
        batch_size = batched_data['s'].size()[0]
        query_emb = batched_data['query_emb'] # batch_size * LLM_size
        query_emb = self.project(query_emb) # batch_size * hidden_size

        target_title_tokens = batched_data['target_title_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_title_masks = batched_data['target_title_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_tokens = batched_data['target_abs_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_masks = batched_data['target_abs_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length

        #with torch.no_grad():
        target_hidden_title_ = self.model(target_title_tokens, target_title_masks).last_hidden_state
        target_hidden_title = self.meanpooling(target_hidden_title_, target_title_masks)
        target_hidden_abs_ = self.model(target_abs_tokens, target_abs_masks).last_hidden_state
        target_hidden_abs = self.meanpooling(target_hidden_abs_, target_abs_masks)
        target_emb = target_hidden_title + target_hidden_abs

        # neighbor modeling
        t_neighbors_node_title_tokens = batched_data['t_nei_n_title_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
        t_neighbors_node_title_masks = batched_data['t_nei_n_title_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
        t_neighbors_node_abs_tokens = batched_data['t_nei_n_abs_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
        t_neighbors_node_abs_masks = batched_data['t_nei_n_abs_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length

        t_nei_hidden_title = self.model(t_neighbors_node_title_tokens, t_neighbors_node_title_masks).last_hidden_state
        t_nei_hidden_title = self.meanpooling(t_nei_hidden_title, t_neighbors_node_title_masks)

        t_nei_hidden_abs = self.model(t_neighbors_node_abs_tokens, t_neighbors_node_abs_masks).last_hidden_state
        t_nei_hidden_abs = self.meanpooling(t_nei_hidden_abs, t_neighbors_node_abs_masks)

        t_nei_hidden = t_nei_hidden_title + t_nei_hidden_abs
        t_nei_hidden = t_nei_hidden.reshape(batch_size, self.args.neigh_num, -1)
        t_nei_hidden = torch.mean(t_nei_hidden, 1)
        
        # get loss
        #target_emb = self.nei_aggregate(torch.cat([target_emb, t_nei_hidden], -1))
        labels = torch.arange(query_emb.shape[0]).cuda()
        g_loss = self.cal_cl_loss(query_emb, target_emb, labels) #self.cal_cl_loss_EU(query_emb, target_emb, labels)
        
        return g_loss
    
    def generate_candidate_emb(self, batched_data):
        batch_size = batched_data['target'].size()[0]
        target_title_tokens = batched_data['target_title_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_title_masks = batched_data['target_title_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_tokens = batched_data['target_abs_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_masks = batched_data['target_abs_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length

        #with torch.no_grad():
        target_hidden_title_ = self.model(target_title_tokens, target_title_masks).last_hidden_state
        target_hidden_title = self.meanpooling(target_hidden_title_, target_title_masks)
        target_hidden_abs_ = self.model(target_abs_tokens, target_abs_masks).last_hidden_state
        target_hidden_abs = self.meanpooling(target_hidden_abs_, target_abs_masks)
        target_emb = target_hidden_title + target_hidden_abs

        # neighbor modeling
        t_neighbors_node_title_tokens = batched_data['t_nei_n_title_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
        t_neighbors_node_title_masks = batched_data['t_nei_n_title_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
        t_neighbors_node_abs_tokens = batched_data['t_nei_n_abs_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
        t_neighbors_node_abs_masks = batched_data['t_nei_n_abs_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length

        t_nei_hidden_title = self.model(t_neighbors_node_title_tokens, t_neighbors_node_title_masks).last_hidden_state
        t_nei_hidden_title = self.meanpooling(t_nei_hidden_title, t_neighbors_node_title_masks)

        t_nei_hidden_abs = self.model(t_neighbors_node_abs_tokens, t_neighbors_node_abs_masks).last_hidden_state
        t_nei_hidden_abs = self.meanpooling(t_nei_hidden_abs, t_neighbors_node_abs_masks)

        t_nei_hidden = t_nei_hidden_title + t_nei_hidden_abs
        t_nei_hidden = t_nei_hidden.reshape(batch_size, self.args.neigh_num, -1)
        t_nei_hidden = torch.mean(t_nei_hidden, 1)

        #target_emb = self.nei_aggregate(torch.cat([target_emb, t_nei_hidden], -1))

        return target_emb
    
    def generate_query_emb(self, batched_data):
        query_emb = batched_data['query_emb'] # batch_size * LLM_size
        query_emb = self.project(query_emb) # batch_size * hidden_size

        return query_emb
    
    def meanpooling(self, embeddings, mask):
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)



class Bert_v2(nn.Module):
    def __init__(self, config, args):
        super(Bert_v2, self).__init__()
        self.args = args
        self.model = AutoModel.from_pretrained(args.pretrained_model)
        self.project = nn.Linear(768, args.hidden_size) #torch.nn.Sequential(nn.Linear(768, args.hidden_size), nn.ReLU(inplace=True), nn.Linear(args.hidden_size, args.hidden_size)) #torch.nn.Sequential( nn.Linear(4096, args.hidden_size), nn.ReLU(inplace=True), nn.Linear(args.hidden_size, 2*args.hidden_size))
        self.nei_aggregate = nn.Linear(2*args.hidden_size, args.hidden_size) #nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.reconstruct_linear = nn.Linear(args.hidden_size, 768)
        self.act = nn.ReLU(inplace=True)

    def cal_cl_loss(self, s_features, t_features, labels):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        logits = logit_scale * self.Cos(s_features, t_features)
        loss_i = F.cross_entropy(logits, labels)
        #loss_t = F.cross_entropy(logits.T, labels)
        ret_loss = loss_i
        return ret_loss
    
    def cal_cl_loss_EU(self, s_features, t_features, labels):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        logits = - logit_scale * self.TransE(s_features.unsqueeze(1), t_features.unsqueeze(0))
        loss_i = F.cross_entropy(logits, labels)
        #loss_t = F.cross_entropy(logits.T, labels)
        ret_loss = loss_i
        return ret_loss

    def Cos(self, h, t):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)

        score = h @ t.t()
        return score
    
    
    def TransE(self, h, t):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)

        score = h - t

        score = torch.norm(score, 2, -1)
        return score
    
    def forward(self, batched_data):
        batch_size = batched_data['s'].size()[0]
        query_emb = batched_data['query_emb'] # batch_size * LLM_size
        query_emb_p = self.project(query_emb) # batch_size * hidden_size

        target_title_tokens = batched_data['target_title_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_title_masks = batched_data['target_title_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_tokens = batched_data['target_abs_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_masks = batched_data['target_abs_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length

        with torch.no_grad():
            target_hidden_title_ = self.model(target_title_tokens, target_title_masks).last_hidden_state
            target_hidden_title = self.meanpooling(target_hidden_title_, target_title_masks)
            target_hidden_abs_ = self.model(target_abs_tokens, target_abs_masks).last_hidden_state
            target_hidden_abs = self.meanpooling(target_hidden_abs_, target_abs_masks)
            target_emb = target_hidden_title + target_hidden_abs

            # neighbor modeling
            t_neighbors_node_title_tokens = batched_data['t_nei_n_title_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_title_masks = batched_data['t_nei_n_title_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_tokens = batched_data['t_nei_n_abs_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_masks = batched_data['t_nei_n_abs_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length

            t_nei_hidden_title = self.model(t_neighbors_node_title_tokens, t_neighbors_node_title_masks).last_hidden_state
            t_nei_hidden_title = self.meanpooling(t_nei_hidden_title, t_neighbors_node_title_masks)

            t_nei_hidden_abs = self.model(t_neighbors_node_abs_tokens, t_neighbors_node_abs_masks).last_hidden_state
            t_nei_hidden_abs = self.meanpooling(t_nei_hidden_abs, t_neighbors_node_abs_masks)

            t_nei_hidden = t_nei_hidden_title + t_nei_hidden_abs
            t_nei_hidden = t_nei_hidden.reshape(batch_size, self.args.neigh_num, -1)
            t_nei_hidden = torch.mean(t_nei_hidden, 1)
        
        target_emb = self.nei_aggregate(torch.cat([target_emb, t_nei_hidden], -1))

        # reconstruct loss
        labels = torch.arange(query_emb.shape[0]).cuda()
        reconstruct_emb = self.reconstruct_linear(self.act(target_emb))
        reconstrct_loss = self.cal_cl_loss(reconstruct_emb, query_emb, labels) #torch.sum((reconstruct_emb - query_emb)**2, -1).mean()

        # get loss
        labels = torch.arange(query_emb.shape[0]).cuda()
        g_loss = self.cal_cl_loss(query_emb_p, target_emb, labels) #self.cal_cl_loss_EU(query_emb, target_emb, labels)
        #g_loss = self.cal_cl_loss(query_emb_p, target_emb+reconstruct_emb, labels) #self.cal_cl_loss_EU(query_emb, target_emb, labels)
        
        return g_loss #+ reconstrct_loss
    
    def generate_candidate_emb(self, batched_data):
        batch_size = batched_data['target'].size()[0]
        target_title_tokens = batched_data['target_title_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_title_masks = batched_data['target_title_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_tokens = batched_data['target_abs_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_masks = batched_data['target_abs_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length

        with torch.no_grad():
            target_hidden_title_ = self.model(target_title_tokens, target_title_masks).last_hidden_state
            target_hidden_title = self.meanpooling(target_hidden_title_, target_title_masks)
            target_hidden_abs_ = self.model(target_abs_tokens, target_abs_masks).last_hidden_state
            target_hidden_abs = self.meanpooling(target_hidden_abs_, target_abs_masks)
            target_emb = target_hidden_title + target_hidden_abs

            # neighbor modeling
            t_neighbors_node_title_tokens = batched_data['t_nei_n_title_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_title_masks = batched_data['t_nei_n_title_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_tokens = batched_data['t_nei_n_abs_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_masks = batched_data['t_nei_n_abs_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length

            t_nei_hidden_title = self.model(t_neighbors_node_title_tokens, t_neighbors_node_title_masks).last_hidden_state
            t_nei_hidden_title = self.meanpooling(t_nei_hidden_title, t_neighbors_node_title_masks)

            t_nei_hidden_abs = self.model(t_neighbors_node_abs_tokens, t_neighbors_node_abs_masks).last_hidden_state
            t_nei_hidden_abs = self.meanpooling(t_nei_hidden_abs, t_neighbors_node_abs_masks)

            t_nei_hidden = t_nei_hidden_title + t_nei_hidden_abs
            t_nei_hidden = t_nei_hidden.reshape(batch_size, self.args.neigh_num, -1)
            t_nei_hidden = torch.mean(t_nei_hidden, 1)

        target_emb = self.nei_aggregate(torch.cat([target_emb, t_nei_hidden], -1))

        # reconstruct
        reconstruct_emb = self.reconstruct_linear(self.act(target_emb))

        return target_emb# + reconstruct_emb
    
    def generate_query_emb(self, batched_data):
        query_emb = batched_data['query_emb'] # batch_size * LLM_size
        query_emb_p = self.project(query_emb) # batch_size * hidden_size

        return query_emb_p
    
    def meanpooling(self, embeddings, mask):
        return embeddings[:, 0, :]
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


class Bert_v3(nn.Module):
    def __init__(self, config, args):
        super(Bert_v3, self).__init__()
        self.args = args
        self.model = AutoModel.from_pretrained(args.pretrained_model)
        self.project = nn.Linear(768, args.hidden_size) #torch.nn.Sequential(nn.Linear(768, args.hidden_size), nn.ReLU(inplace=True), nn.Linear(args.hidden_size, args.hidden_size)) #torch.nn.Sequential( nn.Linear(4096, args.hidden_size), nn.ReLU(inplace=True), nn.Linear(args.hidden_size, 2*args.hidden_size))
        self.nei_aggregate = nn.Linear(2*args.hidden_size, args.hidden_size) #nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.reconstruct_linear = nn.Linear(args.hidden_size, 768)
        self.act = nn.ReLU(inplace=True)

    def cal_cl_loss(self, s_features, t_features, labels):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        logits = logit_scale * self.Cos(s_features, t_features)
        loss_i = F.cross_entropy(logits, labels)
        #loss_t = F.cross_entropy(logits.T, labels)
        ret_loss = loss_i
        return ret_loss
    
    def cal_cl_loss_EU(self, s_features, t_features, labels):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        logits = - logit_scale * self.TransE(s_features.unsqueeze(1), t_features.unsqueeze(0))
        loss_i = F.cross_entropy(logits, labels)
        #loss_t = F.cross_entropy(logits.T, labels)
        ret_loss = loss_i
        return ret_loss

    def Cos(self, h, t):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)

        score = h @ t.t()
        return score
    
    
    def TransE(self, h, t):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)

        score = h - t

        score = torch.norm(score, 2, -1)
        return score
    
    def forward(self, batched_data):
        batch_size = batched_data['s'].size()[0]
        query_emb = batched_data['query_emb'] # batch_size * LLM_size
        query_emb_p = self.project(query_emb) # batch_size * hidden_size

        target_title_tokens = batched_data['target_title_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_title_masks = batched_data['target_title_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_tokens = batched_data['target_abs_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_masks = batched_data['target_abs_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length

        with torch.no_grad():
            target_hidden_title_ = self.model(target_title_tokens, target_title_masks).last_hidden_state
            target_hidden_title = self.meanpooling(target_hidden_title_, target_title_masks)
            target_hidden_abs_ = self.model(target_abs_tokens, target_abs_masks).last_hidden_state
            target_hidden_abs = self.meanpooling(target_hidden_abs_, target_abs_masks)
            target_emb = target_hidden_title + target_hidden_abs

            # neighbor modeling
            t_neighbors_node_title_tokens = batched_data['t_nei_n_title_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_title_masks = batched_data['t_nei_n_title_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_tokens = batched_data['t_nei_n_abs_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_masks = batched_data['t_nei_n_abs_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length

            t_nei_hidden_title = self.model(t_neighbors_node_title_tokens, t_neighbors_node_title_masks).last_hidden_state
            t_nei_hidden_title = self.meanpooling(t_nei_hidden_title, t_neighbors_node_title_masks)

            t_nei_hidden_abs = self.model(t_neighbors_node_abs_tokens, t_neighbors_node_abs_masks).last_hidden_state
            t_nei_hidden_abs = self.meanpooling(t_nei_hidden_abs, t_neighbors_node_abs_masks)

            t_nei_hidden = t_nei_hidden_title + t_nei_hidden_abs
            t_nei_hidden = t_nei_hidden.reshape(batch_size, self.args.neigh_num, -1)
            t_nei_hidden = torch.mean(t_nei_hidden, 1)
        
        target_emb = self.nei_aggregate(torch.cat([target_emb, t_nei_hidden], -1))

        # reconstruct loss
        labels = torch.arange(query_emb.shape[0]).cuda()
        reconstruct_emb = self.reconstruct_linear(self.act(target_emb))
        reconstrct_loss = self.cal_cl_loss(reconstruct_emb, query_emb, labels) #torch.sum((reconstruct_emb - query_emb)**2, -1).mean()

        # get loss
        labels = torch.arange(query_emb.shape[0]).cuda()
        #g_loss = self.cal_cl_loss(query_emb_p, target_emb, labels) #self.cal_cl_loss_EU(query_emb, target_emb, labels)
        g_loss = self.cal_cl_loss(query_emb_p, target_emb+reconstruct_emb, labels) #self.cal_cl_loss_EU(query_emb, target_emb, labels)
        
        return g_loss + reconstrct_loss
    
    def generate_candidate_emb(self, batched_data):
        batch_size = batched_data['target'].size()[0]
        target_title_tokens = batched_data['target_title_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_title_masks = batched_data['target_title_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_tokens = batched_data['target_abs_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_masks = batched_data['target_abs_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length

        with torch.no_grad():
            target_hidden_title_ = self.model(target_title_tokens, target_title_masks).last_hidden_state
            target_hidden_title = self.meanpooling(target_hidden_title_, target_title_masks)
            target_hidden_abs_ = self.model(target_abs_tokens, target_abs_masks).last_hidden_state
            target_hidden_abs = self.meanpooling(target_hidden_abs_, target_abs_masks)
            target_emb = target_hidden_title + target_hidden_abs

            # neighbor modeling
            t_neighbors_node_title_tokens = batched_data['t_nei_n_title_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_title_masks = batched_data['t_nei_n_title_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_tokens = batched_data['t_nei_n_abs_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_masks = batched_data['t_nei_n_abs_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length

            t_nei_hidden_title = self.model(t_neighbors_node_title_tokens, t_neighbors_node_title_masks).last_hidden_state
            t_nei_hidden_title = self.meanpooling(t_nei_hidden_title, t_neighbors_node_title_masks)

            t_nei_hidden_abs = self.model(t_neighbors_node_abs_tokens, t_neighbors_node_abs_masks).last_hidden_state
            t_nei_hidden_abs = self.meanpooling(t_nei_hidden_abs, t_neighbors_node_abs_masks)

            t_nei_hidden = t_nei_hidden_title + t_nei_hidden_abs
            t_nei_hidden = t_nei_hidden.reshape(batch_size, self.args.neigh_num, -1)
            t_nei_hidden = torch.mean(t_nei_hidden, 1)

        target_emb = self.nei_aggregate(torch.cat([target_emb, t_nei_hidden], -1))

        # reconstruct
        reconstruct_emb = self.reconstruct_linear(self.act(target_emb))

        return target_emb + reconstruct_emb
    
    def generate_query_emb(self, batched_data):
        query_emb = batched_data['query_emb'] # batch_size * LLM_size
        query_emb_p = self.project(query_emb) # batch_size * hidden_size

        return query_emb_p
    
    def meanpooling(self, embeddings, mask):
        return embeddings[:, 0, :]
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)



class Bert_v4(nn.Module):
    def __init__(self, config, args):
        super(Bert_v4, self).__init__()
        self.args = args
        self.model = AutoModel.from_pretrained(args.pretrained_model)
        self.project = nn.Linear(768, args.hidden_size) #torch.nn.Sequential(nn.Linear(768, args.hidden_size), nn.ReLU(inplace=True), nn.Linear(args.hidden_size, args.hidden_size)) #torch.nn.Sequential( nn.Linear(4096, args.hidden_size), nn.ReLU(inplace=True), nn.Linear(args.hidden_size, 2*args.hidden_size))
        self.nei_aggregate = nn.Linear(2*args.hidden_size, args.hidden_size) #nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.reconstruct_linear = nn.Linear(args.hidden_size, 768)
        self.act = nn.ReLU(inplace=True)

    def cal_cl_loss(self, s_features, t_features, labels):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        logits = logit_scale * self.Cos(s_features, t_features)
        loss_i = F.cross_entropy(logits, labels)
        #loss_t = F.cross_entropy(logits.T, labels)
        ret_loss = loss_i
        return ret_loss
    
    def cal_cl_loss_EU(self, s_features, t_features, labels):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        logits = - logit_scale * self.TransE(s_features.unsqueeze(1), t_features.unsqueeze(0))
        loss_i = F.cross_entropy(logits, labels)
        #loss_t = F.cross_entropy(logits.T, labels)
        ret_loss = loss_i
        return ret_loss

    def Cos(self, h, t):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)

        score = h @ t.t()
        return score
    
    
    def TransE(self, h, t):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)

        score = h - t

        score = torch.norm(score, 2, -1)
        return score
    
    def forward(self, batched_data):
        batch_size = batched_data['s'].size()[0]
        query_emb = batched_data['query_emb'] # batch_size * LLM_size
        query_emb_p = self.project(query_emb) # batch_size * hidden_size

        target_title_tokens = batched_data['target_title_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_title_masks = batched_data['target_title_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_tokens = batched_data['target_abs_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_masks = batched_data['target_abs_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length

        with torch.no_grad():
            target_hidden_title_ = self.model(target_title_tokens, target_title_masks).last_hidden_state
            target_hidden_title = self.meanpooling(target_hidden_title_, target_title_masks)
            target_hidden_abs_ = self.model(target_abs_tokens, target_abs_masks).last_hidden_state
            target_hidden_abs = self.meanpooling(target_hidden_abs_, target_abs_masks)
            target_emb = target_hidden_title + target_hidden_abs

            # neighbor modeling
            t_neighbors_node_title_tokens = batched_data['t_nei_n_title_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_title_masks = batched_data['t_nei_n_title_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_tokens = batched_data['t_nei_n_abs_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_masks = batched_data['t_nei_n_abs_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length

            t_nei_hidden_title = self.model(t_neighbors_node_title_tokens, t_neighbors_node_title_masks).last_hidden_state
            t_nei_hidden_title = self.meanpooling(t_nei_hidden_title, t_neighbors_node_title_masks)

            t_nei_hidden_abs = self.model(t_neighbors_node_abs_tokens, t_neighbors_node_abs_masks).last_hidden_state
            t_nei_hidden_abs = self.meanpooling(t_nei_hidden_abs, t_neighbors_node_abs_masks)

            t_nei_hidden = t_nei_hidden_title + t_nei_hidden_abs
            t_nei_hidden = t_nei_hidden.reshape(batch_size, self.args.neigh_num, -1)
            t_nei_hidden = torch.mean(t_nei_hidden, 1)
        
        target_emb = self.nei_aggregate(torch.cat([target_emb, t_nei_hidden], -1))

        # reconstruct loss
        labels = torch.arange(query_emb.shape[0]).cuda()
        reconstruct_emb = self.reconstruct_linear(self.act(target_emb))
        reconstrct_loss = self.cal_cl_loss(reconstruct_emb, query_emb, labels) #torch.sum((reconstruct_emb - query_emb)**2, -1).mean()

        # get loss
        labels = torch.arange(query_emb.shape[0]).cuda()
        #g_loss = self.cal_cl_loss(query_emb_p, target_emb, labels) #self.cal_cl_loss_EU(query_emb, target_emb, labels)
        g_loss = self.cal_cl_loss(query_emb_p, reconstruct_emb+target_emb, labels) #self.cal_cl_loss_EU(query_emb, target_emb, labels)
        
        return g_loss# + reconstrct_loss
    
    def generate_candidate_emb(self, batched_data):
        batch_size = batched_data['target'].size()[0]
        target_title_tokens = batched_data['target_title_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_title_masks = batched_data['target_title_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_tokens = batched_data['target_abs_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_masks = batched_data['target_abs_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length

        with torch.no_grad():
            target_hidden_title_ = self.model(target_title_tokens, target_title_masks).last_hidden_state
            target_hidden_title = self.meanpooling(target_hidden_title_, target_title_masks)
            target_hidden_abs_ = self.model(target_abs_tokens, target_abs_masks).last_hidden_state
            target_hidden_abs = self.meanpooling(target_hidden_abs_, target_abs_masks)
            target_emb = target_hidden_title + target_hidden_abs

            # neighbor modeling
            t_neighbors_node_title_tokens = batched_data['t_nei_n_title_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_title_masks = batched_data['t_nei_n_title_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_tokens = batched_data['t_nei_n_abs_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_masks = batched_data['t_nei_n_abs_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length

            t_nei_hidden_title = self.model(t_neighbors_node_title_tokens, t_neighbors_node_title_masks).last_hidden_state
            t_nei_hidden_title = self.meanpooling(t_nei_hidden_title, t_neighbors_node_title_masks)

            t_nei_hidden_abs = self.model(t_neighbors_node_abs_tokens, t_neighbors_node_abs_masks).last_hidden_state
            t_nei_hidden_abs = self.meanpooling(t_nei_hidden_abs, t_neighbors_node_abs_masks)

            t_nei_hidden = t_nei_hidden_title + t_nei_hidden_abs
            t_nei_hidden = t_nei_hidden.reshape(batch_size, self.args.neigh_num, -1)
            t_nei_hidden = torch.mean(t_nei_hidden, 1)

        target_emb = self.nei_aggregate(torch.cat([target_emb, t_nei_hidden], -1))

        # reconstruct
        reconstruct_emb = self.reconstruct_linear(self.act(target_emb))

        return target_emb#+reconstruct_emb
    
    def generate_query_emb(self, batched_data):
        query_emb = batched_data['query_emb'] # batch_size * LLM_size
        query_emb_p = self.project(query_emb) # batch_size * hidden_size

        return query_emb_p
    
    def meanpooling(self, embeddings, mask):
        return embeddings[:, 0, :]
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


class Bert_v5(nn.Module):
    def __init__(self, config, args):
        super(Bert_v5, self).__init__()
        self.args = args
        self.model = AutoModel.from_pretrained(args.pretrained_model)
        self.project = torch.nn.Sequential(nn.Linear(768, args.hidden_size), nn.ReLU(inplace=True), nn.Linear(args.hidden_size, args.hidden_size)) #torch.nn.Sequential( nn.Linear(4096, args.hidden_size), nn.ReLU(inplace=True), nn.Linear(args.hidden_size, 2*args.hidden_size))
        self.nei_aggregate = nn.Linear(2*args.hidden_size, args.hidden_size) #nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.reconstruct_linear = nn.Linear(2*args.hidden_size, 768)
        self.act = nn.ReLU(inplace=True)

    def cal_cl_loss(self, s_features, t_features, labels):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        logits = logit_scale * self.Cos(s_features, t_features)
        loss_i = F.cross_entropy(logits, labels)
        #loss_t = F.cross_entropy(logits.T, labels)
        ret_loss = loss_i
        return ret_loss
    
    def cal_cl_loss_EU(self, s_features, t_features, labels):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        logits = - logit_scale * self.TransE(s_features.unsqueeze(1), t_features.unsqueeze(0))
        loss_i = F.cross_entropy(logits, labels)
        #loss_t = F.cross_entropy(logits.T, labels)
        ret_loss = loss_i
        return ret_loss

    def Cos(self, h, t):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)

        score = h @ t.t()
        return score
    
    def nei_score(self, query_emb, target_emb, t_nei_emb):
        print(query_emb.size())
        print(target_emb.size())
        print(t_nei_emb.size())
        query_emb = query_emb.unsqueeze(0).repeat(len(target_emb), 1)

        target_emb = self.nei_aggregate(torch.cat([target_emb, query_emb], -1))
        nei_emb = self.nei_aggregate(torch.cat([t_nei_emb, query_emb], -1))

        target_emb = F.normalize(target_emb, 2, -1)
        t_nei_emb = F.normalize(t_nei_emb, 2, -1)
        query_emb = F.normalize(query_emb, 2, -1)

        score_1 = torch.sum(query_emb*target_emb, -1)
        score_2 = torch.sum(query_emb*t_nei_emb, -1)

        return score_1 + score_2
    
    
    def TransE(self, h, t):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)

        score = h - t

        score = torch.norm(score, 2, -1)
        return score
    
    def forward(self, batched_data):
        batch_size = batched_data['s'].size()[0]
        query_emb = batched_data['query_emb'] # batch_size * LLM_size
        query_emb_p = self.project(query_emb) # batch_size * hidden_size

        target_title_tokens = batched_data['target_title_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_title_masks = batched_data['target_title_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_tokens = batched_data['target_abs_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_masks = batched_data['target_abs_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length

        with torch.no_grad():
            target_hidden_title_ = self.model(target_title_tokens, target_title_masks).last_hidden_state
            target_hidden_title = self.meanpooling(target_hidden_title_, target_title_masks)
            target_hidden_abs_ = self.model(target_abs_tokens, target_abs_masks).last_hidden_state
            target_hidden_abs = self.meanpooling(target_hidden_abs_, target_abs_masks)
            target_emb = target_hidden_title + target_hidden_abs

            # neighbor modeling
            t_neighbors_node_title_tokens = batched_data['t_nei_n_title_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_title_masks = batched_data['t_nei_n_title_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_tokens = batched_data['t_nei_n_abs_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_masks = batched_data['t_nei_n_abs_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length

            t_nei_hidden_title = self.model(t_neighbors_node_title_tokens, t_neighbors_node_title_masks).last_hidden_state
            t_nei_hidden_title = self.meanpooling(t_nei_hidden_title, t_neighbors_node_title_masks)

            t_nei_hidden_abs = self.model(t_neighbors_node_abs_tokens, t_neighbors_node_abs_masks).last_hidden_state
            t_nei_hidden_abs = self.meanpooling(t_nei_hidden_abs, t_neighbors_node_abs_masks)

            t_nei_hidden = t_nei_hidden_title + t_nei_hidden_abs
            t_nei_hidden = t_nei_hidden.reshape(batch_size, self.args.neigh_num, -1)
            t_nei_hidden = torch.mean(t_nei_hidden, 1)
        
        target_emb = self.nei_aggregate(torch.cat([target_emb, query_emb], -1))
        nei_emb = self.nei_aggregate(torch.cat([t_nei_hidden, query_emb], -1))

        # reconstruct loss
        #labels = torch.arange(query_emb.shape[0]).cuda()
        #reconstruct_emb = self.reconstruct_linear(self.act(target_emb))
        #reconstrct_loss = self.cal_cl_loss(reconstruct_emb, query_emb, labels) #torch.sum((reconstruct_emb - query_emb)**2, -1).mean()

        # get loss
        labels = torch.arange(query_emb.shape[0]).cuda()
        #g_loss = self.cal_cl_loss(query_emb_p, target_emb, labels) #self.cal_cl_loss_EU(query_emb, target_emb, labels)
        g_loss = self.cal_cl_loss(query_emb_p, target_emb, labels) #self.cal_cl_loss_EU(query_emb, target_emb, labels)
        g_loss += self.cal_cl_loss(query_emb_p, nei_emb, labels) #self.cal_cl_loss_EU(query_emb, target_emb, labels)
        
        return g_loss
    
    def generate_candidate_emb(self, batched_data):
        batch_size = batched_data['target'].size()[0]
        target_title_tokens = batched_data['target_title_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_title_masks = batched_data['target_title_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_tokens = batched_data['target_abs_tokens'].reshape(-1, self.args.max_length) # batch_size | args.max_length
        target_abs_masks = batched_data['target_abs_masks'].reshape(-1, self.args.max_length) # batch_size | args.max_length

        with torch.no_grad():
            target_hidden_title_ = self.model(target_title_tokens, target_title_masks).last_hidden_state
            target_hidden_title = self.meanpooling(target_hidden_title_, target_title_masks)
            target_hidden_abs_ = self.model(target_abs_tokens, target_abs_masks).last_hidden_state
            target_hidden_abs = self.meanpooling(target_hidden_abs_, target_abs_masks)
            target_emb = target_hidden_title + target_hidden_abs

            # neighbor modeling
            t_neighbors_node_title_tokens = batched_data['t_nei_n_title_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_title_masks = batched_data['t_nei_n_title_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_tokens = batched_data['t_nei_n_abs_tokens'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length
            t_neighbors_node_abs_masks = batched_data['t_nei_n_abs_masks'].reshape(-1, self.args.max_length) # batch_size * args.neigh_num | args.max_length

            t_nei_hidden_title = self.model(t_neighbors_node_title_tokens, t_neighbors_node_title_masks).last_hidden_state
            t_nei_hidden_title = self.meanpooling(t_nei_hidden_title, t_neighbors_node_title_masks)

            t_nei_hidden_abs = self.model(t_neighbors_node_abs_tokens, t_neighbors_node_abs_masks).last_hidden_state
            t_nei_hidden_abs = self.meanpooling(t_nei_hidden_abs, t_neighbors_node_abs_masks)

            t_nei_hidden = t_nei_hidden_title + t_nei_hidden_abs
            t_nei_hidden = t_nei_hidden.reshape(batch_size, self.args.neigh_num, -1)
            t_nei_hidden = torch.mean(t_nei_hidden, 1)

        return target_emb, t_nei_hidden
    
    def generate_query_emb(self, batched_data):
        query_emb = batched_data['query_emb'] # batch_size * LLM_size
        query_emb_p = self.project(query_emb) # batch_size * hidden_size

        return query_emb_p
    
    def meanpooling(self, embeddings, mask):
        return embeddings[:, 0, :]
        mask = mask.unsqueeze(-1).expand(embeddings.size()).float()
        return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)