import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from transformers import AutoModel, BertLMHeadModel, AutoModelForCausalLM
from transformers.models.bert.modeling_bert import BertEmbeddings, BertPreTrainedModel, BertModel
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType

class retriever(nn.Module):
    def __init__(self, config, args):
        super(retriever, self).__init__()
        self.args = args
        self.project = nn.Linear(1024, args.hidden_size) #torch.nn.Sequential(nn.Linear(768, args.hidden_size), nn.ReLU(inplace=True), nn.Linear(args.hidden_size, args.hidden_size)) #torch.nn.Sequential( nn.Linear(4096, args.hidden_size), nn.ReLU(inplace=True), nn.Linear(args.hidden_size, 2*args.hidden_size))
        self.nei_aggregate = nn.Linear(2*args.hidden_size, args.hidden_size) #nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.reconstruct_linear = nn.Linear(args.hidden_size, 1024)
        self.act = nn.ReLU(inplace=True)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

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
        #h = F.normalize(h, 2, -1)
        #t = F.normalize(t, 2, -1)
        
        score = h@t.t()
        return score
    
    
    def TransE(self, h, t):
        h = F.normalize(h, 2, -1)
        t = F.normalize(t, 2, -1)

        score = h - t

        score = torch.norm(score, 2, -1)
        return score
    
    def forward(self, batched_data):
        batch_size = batched_data['s'].size()[0]
        query_emb = batched_data['query_emb'] # batch_size | LLM_size
        query_emb_p = self.project(query_emb) # batch_size | hidden_size

        target_emb = batched_data['target_emb'] # batch_size | LLM_size
        t_nei_emb = batched_data['target_nei_emb'] # batch_size | args.neigh_num | LLM_size
        t_nei_emb = torch.mean(t_nei_emb, 1)

        target_emb = self.nei_aggregate(torch.cat([target_emb, t_nei_emb], -1))

        # reconstruct loss
        labels = torch.arange(query_emb.shape[0]).cuda()
        reconstruct_emb = self.reconstruct_linear(self.act(target_emb))
        reconstrct_loss = self.cal_cl_loss(reconstruct_emb, query_emb, labels) #torch.sum((reconstruct_emb - query_emb)**2, -1).mean()

        # get loss
        labels = torch.arange(query_emb.shape[0]).cuda()
        #g_loss = self.cal_cl_loss(query_emb_p, target_emb, labels) #self.cal_cl_loss_EU(query_emb, target_emb, labels)
        g_loss = self.cal_cl_loss(query_emb_p, reconstruct_emb+target_emb, labels) #self.cal_cl_loss_EU(query_emb, target_emb, labels)
        
        return g_loss + reconstrct_loss
    
    def generate_candidate_emb(self, batched_data):
        #return batched_data['target_emb']
        batch_size = batched_data['target'].size()[0]
        target_emb = batched_data['target_emb'] # batch_size | LLM_size
        t_nei_emb = batched_data['target_nei_emb'] # batch_size | args.neigh_num | LLM_size
        t_nei_emb = torch.mean(t_nei_emb, 1)

        target_emb = self.nei_aggregate(torch.cat([target_emb, t_nei_emb], -1))

        # reconstruct
        reconstruct_emb = self.reconstruct_linear(self.act(target_emb))

        return target_emb+reconstruct_emb
    
    def generate_query_emb(self, batched_data):
        #return batched_data['query_emb']
        query_emb = batched_data['query_emb'] # batch_size * LLM_size
        query_emb_p = self.project(query_emb) # batch_size * hidden_size

        return query_emb_p