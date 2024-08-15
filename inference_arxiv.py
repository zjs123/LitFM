import argparse
import pickle
import torch
import json
import yaml
import random
import networkx as nx
import numpy as np
from tqdm import tqdm
#from langchain import PromptTemplate
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoConfig, AutoModel, AutoModelForSeq2SeqLM,
                          AutoTokenizer, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, pipeline, AutoModelForCausalLM)

from rouge import Rouge
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from openai import OpenAI
from tqdm import tqdm

"""
Ad-hoc sanity check to see if model outputs something coherent
Not a robust inference platform!
"""

def get_Bleu_score(candidate, reference):
    reference = reference.strip().split(' ')
    candidate = candidate.strip().split(' ')
    #print(reference)
    #print(candidate)
    score = sentence_bleu(reference, candidate)
    return score

def get_ROUGE_score(candidate, reference):
    rouge_score = rouge.get_scores(hyps=candidate, refs=reference)
    return rouge_score[0]["rouge-l"]['p'], rouge_score[0]["rouge-l"]['r'], rouge_score[0]["rouge-l"]['f']

def get_bert_score(candidate, reference):
    P, R, F1 = score([candidate], [reference],lang="en")
    return P, R, F1

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")

def _generate_LP_prompt(data_point: dict, eos_token: str, instruct: bool = False):
    if args.model == 'vicuna':
        Q = '' #instruction
    else:
        Q = ''
    Q = Q + human_instruction[0] + "\n"
    Q = Q + "Here are the information of two papers: \n"
    Q = Q + "Title of Paper A: " + data_point['s_title'] + "\n" if data_point['s_title'] != None else 'Unknown' + "\n"
    Q = Q + "Abstract of Paper A: " + data_point['s_abs'] + "\n" if data_point['s_abs'] != None else 'Unknown' + "\n"
    Q = Q + "Title of Paper B: " + data_point['t_title'] + "\n" if data_point['t_title'] != None else 'Unknown' + "\n"
    Q = Q + "Abstract of Paper B: " + data_point['t_abs'] + "\n" if data_point['t_abs'] != None else 'Unknown' + "\n"
    Q = Q + "Determine if paper A will cite paper B."
    if args.model == 'vicuna':
        Q = Q + " Give me a direct anwser of yes or no by considering the relevance of their topic relevance."

    Q = Q + " Give me a direct anwser of yes or no by considering the relevance of their topic relevance."
    Q = Q + human_instruction[1] + "\n"
    return Q

def _generate_LP_prompt_neighbor(data_point: dict, eos_token: str, instruct: bool = False):
    Q = 'You are a helpful assistant. Your task is to predict the citation relation between two papers. \n Paper 0 will cite Paper 1 if \n  1.The topic of these paper are related.\n 2.The topic of the cited papers of Paper 0 is related to Paper 1.\n 3.The topic of papers who cite Paper 1 is related to Paper 0.'
    Q = Q + 'For example: The topic of Paper 0 is related to inflammatory bowel diseases and the role of carbohydrate-active enzymes in mucus degradation, which is a topic related to gut microbiota and gut health. Paper 1 discusses how the human gut microbiome impacts the serum metabolome and insulin sensitivity, which is also related to gut microbiota and its impact on health. When we look at the papers that have citations with Paper 0 and Paper 1, we can see that there are common themes related to gut microbiota, metabolic disorders, and the impact of microbial communities on human health. This indicates that there is a connection between the topics of Paper 0 and Paper 1 through their citation papers. Therefore, based on the related topics and the common themes in the citation papers, it is likely that Paper 0 will cite Paper 1.'
    Q = human_instruction[0] + "\n"
    Q = Q + "Here are the information of two papers: \n"
    Q = Q + "Title of Paper A: " + data_point['s_title'] + "\n" if data_point['s_title'] != None else 'Unknown' + "\n"
    Q = Q + "Abstract of Paper A: " + data_point['s_abs'] + "\n" if data_point['s_abs'] != None else 'Unknown' + "\n"
    Q = Q + "Other cited Papers of Paper A: \n"
    num = 0
    for title in  data_point['s_nei']:
        Q = Q + str(num) + '. ' + title + ', \n'
        num += 1
    
    Q = Q + "Title of Paper B: " + data_point['t_title'] + "\n" if data_point['t_title'] != None else 'Unknown' + "\n"
    Q = Q + "Abstract of Paper B: " + data_point['t_abs'] + "\n" if data_point['t_abs'] != None else 'Unknown' + "\n"
    Q = Q + "Other Papers that cite Paper B: \n"
    num = 0
    for title in  data_point['t_nei']:
        Q = Q + str(num) + '. ' + title + ', \n'
        num += 1

    Q = Q + "Determine if paper A will cite paper B."
    Q = Q + " Give me a direct answer of yes or no and explain your answer by considering the citation papers of them."

    Q = Q + human_instruction[1] + "\n"
    return Q

def _generate_retrival_prompt(data_point: dict, eos_token: str, instruct: bool = False):
    Q = human_instruction[0] + "\n"
    Q = Q + "Here is the title and abstract of paper A : \n"
    Q = Q + "Title of the Paper A : " + data_point['s_title'] + "\n"
    Q = Q + "Abstract of the Paper A : " + data_point['s_abs'] + "\n"
    Q = Q + "Which of the following papers is more likely to be cited by paper A ? \n"
    for i in range(len(data_point['nei_titles'])):
        Q = Q + str(i) + '. ' + data_point['nei_titles'][i] + '\n'

    if 'vicuna' in args.model:
        Q = Q + "Directly give me the title of the cited paper \n"

    Q = Q + human_instruction[1] + "\n"
    return Q, str(data_point['t_title'])

def _generate_retrival_prompt_neighbor(data_point: dict, eos_token: str, instruct: bool = False):
    Q = 'You are a helpful assistant. Your task is to recommend papers for citing. \n Paper 0 should be cited by Paper 1 if \n 1.The topic of paper 0 and paper 1 are related.\n 2.The topic of other cited papers of Paper 1 is related to the topic of Paper 0.\n'
    Q = Q + human_instruction[0] + "\n"
    Q = Q + "Here are the information of Paper A: \n"
    Q = Q + "Other Papers that cited by Paper A: \n"
    num = 1
    for title in  data_point['retrieval_nei_title']:
        Q = Q + str(num) + ". " + title + ', \n'
        num += 1
    Q = Q + "Title of the Paper A : " + data_point['s_title'] + "\n"
    Q = Q + "Abstract of the Paper A : " + data_point['s_abs'] + "\n"

    Q = Q + "Which of the following papers is more likely to be cited by Paper A ?\n"

    for i in range(len(data_point['nei_titles'])):
        Q = Q + str(i) + '. ' + data_point['nei_titles'][i] + '\n'

    Q = Q + human_instruction[1] + "\n"
    return Q, str(data_point['t_title'])


def _generate_abstrat_2_title_prompt(data_point: dict, eos_token: str, instruct: bool = False):
    Q = ''   
    Q = Q + human_instruction[0] + "\n"
    Q = Q + "Here is the abstract of paper A, please generate the title of paper A. \n"
    Q = Q + "Abstract of Paper A: " + data_point['abs'] + "\n"

    Q = Q + human_instruction[1] + "\n"
    #Q = Q + "Title of Paper A: "# + data_point['title'].strip().split()[0]

    return Q

def _generate_abstrat_2_title_prompt_neighbor(data_point: dict, eos_token: str, instruct: bool = False):
    Q = ''   
    Q = Q + human_instruction[0] + "\n"
    Q = Q + "Here is the abstract of paper A. \n"
    Q = Q + "Abstract of Paper A: " + data_point['abs'] + "\n"

    Q = Q + "Title of other papers that cited by Paper A: \n"
    i = 1
    for title in  data_point['retrieval_nei_titles']:
        Q = Q + str(i) + '. ' + title + '\n'
        i += 1

    Q = Q + "Reference the above titles to generate the title of paper A.\n"
    Q = Q + human_instruction[1] + "\n"
    Q = Q + "Title of Paper A: "# + data_point['title'].strip().split()[0]

    return Q

def _generate_sentence_prompt(data_point: dict, eos_token: str, instruct: bool = False):
    Q = human_instruction[0] + "\n"
    Q = Q + "Here are the information of two papers: \n"
    Q = Q + "Title of Paper A: " + data_point['s_title'] + '\n' if data_point['s_title'] != None else 'Unknown' + "\n"
    Q = Q + "Abstract of Paper A: " + data_point['s_abs'] + '\n' if data_point['s_abs'] != None else 'Unknown' + "\n"
    Q = Q + "Title of Paper B: " + data_point['t_title'] + '\n' if data_point['t_title'] != None else 'Unknown' + "\n"
    Q = Q + "Abstract of Paper B: " + data_point['t_abs'] + '\n' if data_point['t_abs'] != None else 'Unknown' + "\n"

    Q = Q + "Generate the citation sentence of Paper A cites paper B."

    Q = Q + human_instruction[1] + "\n"
    Q = Q + data_point['sentence'].strip().split()[0]

    return Q

def _generate_sentence_prompt_neighbor(data_point: dict, eos_token: str, instruct: bool = False):
    Q = human_instruction[0] + "\n"
    Q = Q + "Here are the information of two papers: \n"
    Q = Q + "Title of Paper A: " + data_point['s_title'] + '\n' if data_point['s_title'] != None else 'Unknown' + "\n"
    Q = Q + "Abstract of Paper A: " + data_point['s_abs'] + '\n' if data_point['s_abs'] != None else 'Unknown' + "\n"
    Q = Q + "Title of Paper B: " + data_point['t_title'] + '\n' if data_point['t_title'] != None else 'Unknown' + "\n"
    Q = Q + "Abstract of Paper B: " + data_point['t_abs'] + '\n' if data_point['t_abs'] != None else 'Unknown' + "\n"

    Q = Q + "Here are some citation sentences from other papers to paper B: \n"
    for i in range(len(data_point['t_nei_sentence'])):
        #Q = Q + str(i) + ". Paper: " + data_point['nei_title'][i] + " Citation sentence: " + data_point['nei_sentence'][i] + '\n'
        Q = Q + str(i) + "." + data_point['t_nei_sentence'][i] + '\n'
    Q = Q + "Generate the citation sentence of Paper A cites paper B based on the provided citation sentences. \n"
    #Q = Q + "Generate the citation sentence of Paper A cites paper B."

    Q = Q + human_instruction[1] + "\n"
    Q = Q + data_point['sentence'].strip().split()[0]

    return Q

def _generate_abstrat_completion_prompt(data_point: dict, eos_token: str, instruct: bool = False):
    split_abs = data_point['abs'][: int(0.1*len(data_point['abs']))]
    Q = human_instruction[0] + "\n"
    Q = Q + "Here is the title of paper A : " + data_point['title'] + "\n"
    Q = Q + "Please complete the abstract of paper A : " + split_abs + '\n'

    Q = Q + human_instruction[1] + "\n"
    Q = Q + "Abstract of Paper A: " + split_abs

    return Q

def _generate_abstrat_completion_prompt_neighbor(data_point: dict, eos_token: str, instruct: bool = False):
    split_abs = data_point['abs'][: int(0.1*len(data_point['abs']))]
    Q = human_instruction[0] + "\n"
    Q = Q + "Here is the title of paper A : " + data_point['title'] + "\n"
    
    Q = Q + "Here are abstracts of other papers related to paper A: \n"
    for i in range(len(data_point['nei_abs'])):
        Q = Q + str(i) + "." + data_point['nei_abs'][i] + '\n'
    Q = Q + "Generate the abstract of Paper A based on the provided abstracts. \n"

    Q = Q + human_instruction[1] + "\n"
    Q = Q + "Abstract of Paper A: " + split_abs

    return Q

def _generate_summary_prompt(data_point: dict, eos_token: str, instruct: bool = False):

    # few-shot example
    #Q = human_instruction[1] + "\n"
    #Q = Q + "Related work: Oxidative stress has been recognized as a critical factor in the development of dry eye disease [2]. This condition is characterized by inflammation of the lacrimal gland and damage to the lipid layer of the tear film, leading to reduced tear film quality and instability [3]. The accumulation of reactive oxygen species (ROS) plays a significant role in oxidative stress, contributing to the progression of dry eye disease [3]. Therefore, understanding the mechanisms underlying oxidative stress and its impact on dry eye disease is crucial for developing effective treatments. One potential approach involves targeting the downstream effects of oxidative stress, such as inflammation, through the use of natural compounds with antioxidant properties [1]. Purple sweet potato powder contains anthocyanins, which have been shown to possess strong antioxidant activity [4]. By incorporating purple sweet potato powder into a dietary supplement, it may be possible to reduce oxidative stress and alleviate symptoms of dry eye disease."
    #Q = Q + "\n"

    # prompt
    Q = human_instruction[0] + "\n"
    Q = Q + "Here is the title and abstract of paper A : \n"
    Q = Q + "Title of the Paper A : " + data_point['s_title'] + "\n"
    Q = Q + "Abstract of the Paper A : " + data_point['s_abs'] + "\n"
    Q = Q + "Please generate the related work section of this paper. \n"
    Q = Q + "Note that each citation should be indicated. \n"
    Q = Q + human_instruction[1] + "\n"
    return Q

def _generate_summary_cluster_prompt(data_point: dict, eos_token: str, instruct: bool = False):

    Q = human_instruction[0] + "\n"
    Q = Q + "Here is the title and abstract of paper A : \n"
    Q = Q + "Title of the Paper A : " + data_point['s_title'] + "\n"
    Q = Q + "Abstract of the Paper A : " + data_point['s_abs'] + "\n"
    Q = Q + "Please generate the related work section of this paper based on the following information. \n"
    
    for i in range(len(data_point['nei_title'])):
        Q = Q + str(i+1) + ". Paper: " + data_point['nei_title'][i]  + '\n'# + " Citation sentence: " + data_point['nei_sentence'][i] + '\n'
        #Q = Q + str(i) + "." + " Citation sentence: " + data_point['nei_sentence'][i] + '\n'
    #Q = Q + "Please organize these citation sentences as a paragraph. \n"
    #Q = Q + "Please organize these citation sentences as a paragraph which can be used as a related work section. \n"
    #Q = Q + "Please first generate the citation sentences of Paper A cites each of the above paper. \n"
    Q = Q + "You should first generate citation sentences for each of the above papers.\n"
    Q = Q + "And then you should devide the above "+ str(len(data_point['nei_title'])) +" citation sentences as several groups, and then summary each group as a paragraph, finally organize these flowing paragraphs as a related work section. Each citation should be indicated. \n" # vicuna7b without lora
    #Q = Q + "Each citation should be indicated."
    Q = Q + human_instruction[1] + "\n"
    Q = Q + "Related work: " # vicuna7b without lora
    #if args.model == 'vicuna':
    #    Q = Q + "Related work section: \n" # vicuna7b without lora

    return Q

def get_llm_response(prompt):
    raw_output = pipe(prompt)
    return raw_output

def test_sentence(model_name):
    Bert_p_list = []
    Bert_r_list = []
    Bert_f_list = []

    result_dict = {}
    # pos test
    for i in range(100): #range(len(test_data)):
        source, target = test_data[i][0], test_data[i][1]
        source_title, source_abs = raw_id_2_tile_abs[source]
        target_title, target_abs = raw_id_2_tile_abs[target]
        
        s_nei = list(nx.all_neighbors(raw_graph, source))
        s_nei_list = list(set(s_nei) - set([source]) - set([target]))[:10]
        s_nei_titles = [raw_id_2_tile_abs[i][0] for i in s_nei_list]

        t_nei = list(nx.all_neighbors(raw_graph, target))
        t_nei_list = list(set(t_nei) - set([source]) - set([target]))[:10]
        t_nei_titles = [raw_id_2_tile_abs[i][0] for i in t_nei_list]

        t_nei_sentence = []
        for i in range(len(t_nei_list)):
            tmp_sentence = raw_id_pair_2_sentence[(t_nei_list[i], target)] if (t_nei_list[i], target) in raw_id_pair_2_sentence.keys() else ''
            if len(tmp_sentence) != 0:
                t_nei_sentence.append(tmp_sentence)

        citation_sentence = raw_id_pair_2_sentence[(source, target)] if (source, target) in raw_id_pair_2_sentence.keys() else raw_id_pair_2_sentence[(target, source)]
        
        datapoint = {'s_title':source_title, 's_abs':source_abs, 't_title':target_title, 't_abs':target_abs, 's_nei':s_nei_titles, 't_nei':t_nei_titles, 't_nei_sentence':t_nei_sentence, 'sentence': citation_sentence}
        if 'GPT' in model_name:
            prompt = _generate_sentence_prompt_neighbor(datapoint, '') #_generate_sentence_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_sentence_prompt(datapoint, tokenizer.eos_token)
            ans = gpt_35_api([{'role': 'user','content': prompt}])
        if 'vicuna' in model_name:
            prompt = _generate_sentence_prompt_neighbor(datapoint, '') #_generate_sentence_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_sentence_prompt(datapoint, tokenizer.eos_token)
            ans = get_llm_response(prompt)[0]['generated_text']
        if 'lora' in model_name:
            prompt = _generate_sentence_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_sentence_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_sentence_prompt(datapoint, tokenizer.eos_token)
            ans = get_llm_response(prompt)[0]['generated_text']
        res = ans.strip().split(human_instruction[1]+'\n')[-1]
        res = res.strip().split('.')[0]

        result_dict[(source, target)] = [source_title, source_abs, target_title, target_abs, citation_sentence, res]

        print(ans)
        print(res)
        print(citation_sentence)
        
        Bert_p, Bert_r, Bert_f = get_bert_score(res, citation_sentence)
        Bert_p_list.append(Bert_p)
        Bert_r_list.append(Bert_r)
        Bert_f_list.append(Bert_f)
        print([len(Bert_p_list), np.mean(Bert_p_list), np.mean(Bert_r_list), np.mean(Bert_f_list)])
    
    #pickle.dump(result_dict, open("sentence_example_it_CiteGRAM.pkl", 'wb'))

def test_LP(model_name):
    result_list = []
    # pos test 1919/2000
    
    for i in range(500): #range(len(test_data)):
        source, target = test_data[i][0], test_data[i][1]
        source_title, source_abs = raw_id_2_tile_abs[source]
        target_title, target_abs = raw_id_2_tile_abs[target]
        
        s_nei = list(nx.all_neighbors(raw_graph, source))
        s_nei_list = list(set(s_nei) - set([source]) - set([target]))[:5]
        s_nei_titles = [raw_id_2_tile_abs[i][0] for i in s_nei_list]

        t_nei = list(nx.all_neighbors(raw_graph, target))
        t_nei_list = list(set(t_nei) - set([source]) - set([target]))[:5]
        t_nei_titles = [raw_id_2_tile_abs[i][0] for i in t_nei_list]
        
        datapoint = {'s_title':source_title, 's_abs':source_abs, 't_title':target_title, 't_abs':target_abs, 's_nei':s_nei_titles, 't_nei':t_nei_titles, 'label':'yes'}
        if 'GPT' in model_name:
            prompt = _generate_LP_prompt_neighbor(datapoint, '') #_generate_abstrat_2_title_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt(datapoint, tokenizer.eos_token)
            ans = gpt_35_api([{'role': 'user','content': prompt}])
        if 'vicuna' in model_name:
            prompt = _generate_LP_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt(datapoint, tokenizer.eos_token)
            ans = get_llm_response(prompt)[0]['generated_text']
        if 'lora' in model_name:
            prompt = _generate_LP_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt(datapoint, tokenizer.eos_token)
            ans = get_llm_response(prompt)[0]['generated_text']
        
        res = ans.strip().split(human_instruction[1])[-1]
        
        print(ans)
        print(res)
        if 'yes' in res or 'Yes' in res:
            result_list.append(1)
        else:
            result_list.append(0)
        print([sum(result_list), len(result_list)])
    
     # neg test
    for i in range(500): #range(len(test_data)):
        source, target = test_data[i][0], random.sample(list(graph_data.nodes()), 1)[0]
        source_title, source_abs = raw_id_2_tile_abs[source]
        target_title, target_abs = raw_id_2_tile_abs[target]

        s_nei = list(nx.all_neighbors(raw_graph, source))
        s_nei_list = list(set(s_nei) - set([source]) - set([target]))[:5]
        s_nei_titles = [raw_id_2_tile_abs[i][0] for i in s_nei_list]

        try:
            t_nei = list(nx.all_neighbors(raw_graph, target))
        except:
            t_nei = []
        t_nei_list = list(set(t_nei) - set([source]) - set([target]))[:5]
        t_nei_titles = [raw_id_2_tile_abs[i][0] for i in t_nei_list]

        datapoint = {'s_title':source_title, 's_abs':source_abs, 't_title':target_title, 't_abs':target_abs, 's_nei':s_nei_titles, 't_nei':t_nei_titles, 'label':'no'}
        if 'GPT' in model_name:
            prompt = _generate_LP_prompt_neighbor(datapoint, '') #_generate_abstrat_2_title_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt(datapoint, tokenizer.eos_token)
            ans = gpt_35_api([{'role': 'user','content': prompt}])
        if 'vicuna' in model_name:
            prompt = _generate_LP_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt(datapoint, tokenizer.eos_token)
            ans = get_llm_response(prompt)[0]['generated_text']
        if 'lora' in model_name:
            prompt = _generate_LP_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt(datapoint, tokenizer.eos_token)
            ans = get_llm_response(prompt)[0]['generated_text']
        res = ans.strip().split(human_instruction[1])[-1]
        
        if 'No' in res or 'no' in res:
            result_list.append(1)
        else:
            result_list.append(0)
        print([sum(result_list), len(result_list)])
    
    print([sum(result_list), len(result_list)])


def test_retrival_e(model_name):
    result_list = []
    # pos test
    for i in range(1000): #range(len(test_data)):
        source, target = test_data[i][0], test_data[i][1]
        source_title, source_abs = raw_id_2_tile_abs[source]
        target_title, target_abs = raw_id_2_tile_abs[target]

        neighbors = list(nx.all_neighbors(raw_graph, source))
        sample_node_list = list(set(raw_graph.nodes()) - set(neighbors) - set([source]) - set([target]))
        sampled_neg_nodes = random.sample(sample_node_list, 5) + [target]
        random.shuffle(sampled_neg_nodes)

        retrieval_nei = node_id_2_retrieval_papers[source] #list(nx.all_neighbors(raw_graph, source)) #node_id_2_retrieval_papers[source] # neighbors
        retrieval_nei_list = list(set(retrieval_nei) - set([source]) - set([target]))[:3]
        retrieval_nei_titles = [raw_id_2_tile_abs[i][0] for i in retrieval_nei_list]
        
        datapoint = {'s_title':source_title, 's_abs':source_abs, 't_title':target_title, 'nei_titles':[raw_id_2_tile_abs[node][0] for node in sampled_neg_nodes], 'retrieval_nei_title':retrieval_nei_titles}
        if 'GPT' in model_name:
            prompt, label = _generate_retrival_prompt_neighbor(datapoint, '') #_generate_abstrat_2_title_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt(datapoint, tokenizer.eos_token)
            ans = gpt_35_api([{'role': 'user','content': prompt}])
        if 'vicuna' in model_name:
            prompt, label = _generate_retrival_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt(datapoint, tokenizer.eos_token)
            ans = get_llm_response(prompt)[0]['generated_text']
        if 'lora' in model_name:
            prompt, label = _generate_retrival_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt(datapoint, tokenizer.eos_token)
            ans = get_llm_response(prompt)[0]['generated_text']
        
        res = ans.strip().split(human_instruction[1])[-1]
        
        print(ans)
        print(target_title)
        print(res)
        if target_title[:int(len(target_title)*0.5)] in res or res in target_title or 'Paper'+str(label) in res:
            result_list.append(1)
        else:
            result_list.append(0)
        print([sum(result_list), len(result_list)])
    
    print([sum(result_list), len(result_list)])

def test_title_generate(model_name):
    result_dict = {}
    Bleu_list = []
    ROUGE_p_list = []
    ROUGE_r_list = []
    ROUGE_f_list = []
    Bert_p_list = []
    Bert_r_list = []
    Bert_f_list = []
    # pos test
    for i in range(100): #range(len(test_data)):
        source, target = test_data[i][0], test_data[i][1]
        title, abstract = raw_id_2_tile_abs[source]
        if title == None or abstract == None:
            continue

        retrieval_nei = node_id_2_retrieval_papers[source] #list(nx.all_neighbors(raw_graph, source)) #node_id_2_retrieval_papers[source]
        retrieval_nei_list = list(set(retrieval_nei) - set([source]) - set([target]))[:5]
        retrieval_nei_titles = [raw_id_2_tile_abs[i][0] for i in retrieval_nei_list]
        
        datapoint = {'title':title, 'abs':abstract, 'retrieval_nei_titles':retrieval_nei_titles}
        if 'GPT' in model_name:
            prompt = _generate_abstrat_2_title_prompt_neighbor(datapoint, '') #_generate_abstrat_2_title_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt(datapoint, tokenizer.eos_token)
            ans = gpt_35_api([{'role': 'user','content': prompt}])
        if 'vicuna' in model_name:
            prompt = _generate_abstrat_2_title_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt(datapoint, tokenizer.eos_token)
            ans = get_llm_response(prompt)[0]['generated_text']
        if 'lora' in model_name:
            prompt = _generate_abstrat_2_title_prompt(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt(datapoint, tokenizer.eos_token)
            ans = get_llm_response(prompt)[0]['generated_text']
        res = ans.strip().split(human_instruction[1]+'\n')[-1]
        res = res.strip().split('.')[0]

        result_dict[source] = [title, abstract, res]
        
        print(ans)
        print(res)
        print(title)
        
        Bert_p, Bert_r, Bert_f = get_bert_score(res, title)
        Bert_p_list.append(Bert_p)
        Bert_r_list.append(Bert_r)
        Bert_f_list.append(Bert_f)
        print([len(Bert_p_list), np.mean(Bert_p_list), np.mean(Bert_r_list), np.mean(Bert_f_list)])

    
    #pickle.dump(result_dict, open("title_example_it_CiteGRAM.pkl", 'wb'))

def test_abs_completion(model_name):
    result_dict = {}
    Bleu_list = []
    ROUGE_p_list = []
    ROUGE_r_list = []
    ROUGE_f_list = []
    Bert_p_list = []
    Bert_r_list = []
    Bert_f_list = []
    # pos test
    for i in range(1000): #range(len(test_data)):
        source, target = test_data[i][0], test_data[i][1]
        title, abstract = raw_id_2_tile_abs[source]
        if title == None or abstract == None:
            continue

        retrieval_nei = node_id_2_retrieval_papers[source] #list(nx.all_neighbors(raw_graph, source)) #node_id_2_retrieval_papers[source]
        retrieval_nei_list = list(set(retrieval_nei) - set([source]) - set([target]))[:5]
        retrieval_nei_abs = [raw_id_2_tile_abs[i][1] for i in retrieval_nei_list]
        
        datapoint = {'title':title, 'abs':abstract, 'nei_abs':retrieval_nei_abs}
        
        if 'GPT' in model_name:
            prompt = _generate_abstrat_completion_prompt_neighbor(datapoint, '') #_generate_abstrat_completion_prompt(datapoint, tokenizer.eos_token)
            ans = gpt_35_api([{'role': 'user','content': prompt}])
        if 'vicuna' in model_name:
            prompt = _generate_abstrat_completion_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt(datapoint, tokenizer.eos_token)
            ans = get_llm_response(prompt)[0]['generated_text']
        if 'lora' in model_name:
            prompt = _generate_abstrat_completion_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt_neighbor(datapoint, tokenizer.eos_token) #_generate_abstrat_2_title_prompt(datapoint, tokenizer.eos_token)
            ans = get_llm_response(prompt)[0]['generated_text']
        
        res = ans.strip()#.split(human_instruction[1]+'\n')[-1]

        result_dict[source] = [title, abstract, res]
        
        print(ans)
        print(res)
        print(abstract)
        
        Bert_p, Bert_r, Bert_f = get_bert_score(res, abstract)
        Bert_p_list.append(Bert_p)
        Bert_r_list.append(Bert_r)
        Bert_f_list.append(Bert_f)
        print([len(Bert_p_list), np.mean(Bert_p_list), np.mean(Bert_r_list), np.mean(Bert_f_list)])

    
    #pickle.dump(result_dict, open("abs_example_it_CiteGRAM.pkl", 'wb'))

def test_summary(model_name):
    result_dict = {}
    Bert_p_list = []
    Bert_r_list = []
    Bert_f_list = []

    ROUGE_p_list = []
    ROUGE_r_list = []
    ROUGE_f_list = []

    recall_score = []

    real_len, generate_len = [], []
    real_cite, generate_cite = [], []
    real_para, generate_para = [], []

    sentence_without_citations_ratio = []
    sentence_without_citations_ratio_real = []
    # pos test
    for i in list(related_work_data.keys())[:500]:
        source = i
        related_work_ground_truth = related_work_data[i][0]
        target_papers = related_work_data[i][1]
        if source not in raw_id_2_tile_abs.keys():
            continue
        source_title, source_abs = raw_id_2_tile_abs[source]
        
        s_nei = list(target_papers)
        s_nei_list = list(set(s_nei) - set([source])) + list(random.sample(list(raw_id_2_tile_abs.keys()), 5))
        s_nei_titles = random.sample(s_nei_list, len(s_nei))
        s_nei_titles = [raw_id_2_tile_abs[i][0] for i in s_nei_list]
        s_nei_sentences = []
        for sampled_nei in s_nei_list:
            try:
                citation_sentence = raw_id_pair_2_sentence[(source, sampled_nei)] if (source, sampled_nei) in raw_id_pair_2_sentence.keys() else raw_id_pair_2_sentence[(sampled_nei, source)]
                s_nei_sentences.append(citation_sentence)
            except:
                s_nei_sentences.append('')
        if len(s_nei_sentences) >= 5 and len(s_nei_sentences) <= 20: #if len(s_nei_sentences) >= 10 and len(s_nei_sentences) <= 20:
            datapoint = {'s_title':source_title, 's_abs':source_abs, 'nei_title':s_nei_titles, 'nei_sentence': s_nei_sentences}
            if 'GPT' in model_name:
                prompt = _generate_summary_prompt(datapoint, '') #_generate_abstrat_completion_prompt(datapoint, tokenizer.eos_token)
                ans = gpt_35_api([{'role': 'user','content': prompt}])
            if 'vicuna' in model_name:
                prompt = _generate_summary_prompt(datapoint, tokenizer.eos_token) #_generate_abstrat_completion_prompt(datapoint, tokenizer.eos_token)
                ans = get_llm_response(prompt)[0]['generated_text']
            if 'lora' in model_name:
                prompt = _generate_summary_cluster_prompt(datapoint, tokenizer.eos_token) #_generate_abstrat_completion_prompt(datapoint, tokenizer.eos_token)
                ans = get_llm_response(prompt)[0]['generated_text']
            res = ans.strip().split(human_instruction[1]+'\n')[-1]

            result_dict[source] = [source_title, source_abs, list(target_papers), s_nei_list, related_work_ground_truth, res]

            print(prompt)
            print(res)

            count = 0
            for i in range(len(s_nei_sentences)):
                notation = "[" + str(i+1) + "]"
                notation_2 = str(i+1) + "]"
                if notation in res or notation_2 in res:
                    count += 1
            count += len(res.split('\cite'))-1

            try:
                Bert_p, Bert_r, Bert_f = get_bert_score(res, related_work_ground_truth) #get_bert_score(res, related_work_ground_truth)
                Bert_p_list.append(Bert_p)
                Bert_r_list.append(Bert_r)
                Bert_f_list.append(Bert_f)

                ROUGE_p, ROUGE_r, ROUGE_f = get_ROUGE_score(res, related_work_ground_truth) #get_bert_score(res, related_work_ground_truth)
                ROUGE_p_list.append(ROUGE_p)
                ROUGE_r_list.append(ROUGE_r)
                ROUGE_f_list.append(ROUGE_f)
                if len(s_nei_sentences) != 0:
                    recall_score.append(float(count)/float(len(s_nei_sentences)))
                
                real_len.append(len(related_work_ground_truth.split(' ')))
                generate_len.append(len(res.split(' ')))

                real_cite.append(len(related_work_ground_truth.split('\cite')))
                generate_cite.append(count)

                real_para.append(len(related_work_ground_truth.split('\n\n')))
                generate_para.append(len(res.split('\n\n')))

                citation_sentence_count = 0
                all_sentences = 0
                for i in res.split('\n\n'):
                    if len(i.split(' ')) >= 20:
                        if ('[' in i and ']' in i ) or '\cite' in i:
                            citation_sentence_count += 1
                        all_sentences += 1
                sentence_without_citations_ratio.append(citation_sentence_count/(all_sentences+0.01))

                citation_sentence_count_real = 0
                all_sentences_real = 0
                for i in related_work_ground_truth.split('\n\n'):
                    if len(i.split(' ')) >= 20:
                        if ('[' in i and ']' in i ) or '\cite' in i:
                            citation_sentence_count_real += 1
                        all_sentences_real += 1
                        
                sentence_without_citations_ratio_real.append(citation_sentence_count_real/(all_sentences_real+0.01))

                print([len(Bert_p_list), np.mean(Bert_p_list), np.mean(Bert_r_list), np.mean(Bert_f_list), np.mean(recall_score), np.mean(sentence_without_citations_ratio), np.mean(sentence_without_citations_ratio_real)])
                print([len(ROUGE_p_list), np.mean(ROUGE_p_list), np.mean(ROUGE_r_list), np.mean(ROUGE_f_list)])
                print([np.mean(real_len), np.mean(generate_len)])
                print([np.mean(real_cite), np.mean(generate_cite)])
                print([np.mean(real_para), np.mean(generate_para)])

            except:
                pass
    pickle.dump(result_dict, open("summary_example_it_vicuna.pkl", 'wb'))

def gpt_35_api(messages: list):
    """为提供的对话消息创建新的回答

    Args:
        messages (list): 完整的对话消息
    """
    completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages) # gpt-3.5-turbo
    message = completion.choices[0].message.content
    return message.strip()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config_path", help="Path to the config YAML file")
    parser.add_argument("-model", help="Path to the config YAML file")
    parser.add_argument("-prompt_num", help="Path to the config YAML file", default = 1)
    args = parser.parse_args()

    config = read_yaml_file(args.config_path)

    print("Load model")
    if args.model == 'vicuna':
        model_path = config["base_model"]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, load_in_8bit=True)
    if args.model == 'lora': 
        model_path = config["base_model"]
        if 'llama' in model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, load_in_8bit=True)
            tokenizer.model_max_length = 512
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", load_in_8bit=True)

        adapter_save_path = f"{config['model_output_dir']}/{config['model_name']}_adapter_abs2title_abscomp_arxiv" # LP_retrieval_sentence_arxiv # abs2title_abscomp_arxiv
        model = PeftModel.from_pretrained(base_model, adapter_save_path)
        model = model.merge_and_unload()

    if 'GPT' in args.model:
        client = OpenAI(
            api_key=api_key,
            base_url=""
        )
        human_instruction = ['### HUMAN:', '### RESPONSE:']
    else:
        if 'mistral' in model_path:
            pipe = pipeline(
                "text-generation",
                model=model, 
                tokenizer=tokenizer, 
                #max_length=6096,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                max_new_tokens=100,
            )
            human_instruction = ['[INST]', '[/INST]']
        
        elif 'llama' in model_path:
            pipe = pipeline(
                "text-generation",
                model=model, 
                tokenizer=tokenizer, 
                max_length=4096,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                #max_new_tokens=6096,
            )
            human_instruction = ['### HUMAN:', '### RESPONSE:']
        else:
            pipe = pipeline(
                "text-generation",
                model=model, 
                tokenizer=tokenizer, 
                max_length=4096,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                #max_new_tokens=6096,
            )
            human_instruction = ['### HUMAN:', '### RESPONSE:']
    rouge = Rouge()

    graph_data = nx.read_gexf("cs_it_processed_cite_removed.gexf", node_type=None, relabel=False, version='1.2draft')
    test_data = pickle.load(open('test_arxiv.pkl', 'rb'))
    related_work_data = pickle.load(open('related_work_ground_truth_arxiv_it.pkl', 'rb'))
    raw_graph = graph_data
    print([len(graph_data.nodes()), len(graph_data.edges())])

    with open("title_abstracts_cs_it.json",'r', encoding='UTF-8') as f:
        raw_id_2_tile_abs = dict()
        tmp_id_2_tile_abs = json.load(f)
        for paper_id in tmp_id_2_tile_abs:
            title = tmp_id_2_tile_abs[paper_id]['Title']
            abstract = tmp_id_2_tile_abs[paper_id]['Abstract']
            raw_id_2_tile_abs[paper_id] = [title, abstract]


    raw_id_pair_2_sentence = dict()
    for edge in list(graph_data.edges()):
        sentence = graph_data.edges()[edge]['sentence']
        raw_id_pair_2_sentence[edge] = sentence
    #print(raw_id_pair_2_sentence)
    edge_list = list(raw_graph.edges())[:args.prompt_num]

    args.dataset_name = 'arxiv_it'
    retrieval_results = pickle.load(open("navigator/retrieval_result_all_test_nodes"+args.dataset_name+".pkl","rb"))
    node_id_2_retrieval_papers = {}
    for i in range(len(retrieval_results[0])):
        query = retrieval_results[0][i]
        pred = retrieval_results[1][i]
        node_id_2_retrieval_papers[query] = pred