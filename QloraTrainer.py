import json
import torch
import random
import pickle
import transformers
import networkx as nx
from tqdm import tqdm
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict
from peft import (LoraConfig, PeftModel, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer)

# Define a custom data collator
class CustomDataCollator:
    def __call__(self, batch):
        input_ids, attention_mask, labels = zip(*batch)
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

class QloraTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.tokenizer = None
        self.base_model = None
        self.adapter_model = None
        self.merged_model = None
        self.human_instruction = ['### HUMAN:', '### RESPONSE:']

    def load_base_model(self):
        model_id = self.config["base_model"]
        print(model_id)

        if 'llama' in model_id:
            bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.bfloat16
            )
            print('load llama 3')
            access_token = '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'
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
            tokenizer.model_max_length = 512
            tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, torch_dtype=torch.bfloat16, device_map={"":0}, token=access_token)
        else:
            bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.bfloat16
            )
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
            if not tokenizer.pad_token:
                # Add padding token if missing, e.g. for llama tokenizer
                #tokenizer.pad_token = tokenizer.eos_token  # https://github.com/huggingface/transformers/issues/22794
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        self.tokenizer = tokenizer
        self.base_model = model

    def load_adapter_model(self, adapter_path: str):
        """ Load pre-trained lora adapter """
        self.adapter_model = PeftModel.from_pretrained(self.base_model, adapter_path)

    def train(self):
        # Set up lora config or load pre-trained adapter
        config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=self.config["target_modules"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(self.base_model, config)
        self._print_trainable_parameters(model)

        print("Start data preprocessing")
        # TODO: Expand this to cover more dataset types and processing patterns
        data = self._process_data() #self._process_vicuna_data()

        print("Start training")
        trainer = transformers.Trainer(
            model=model,
            train_dataset=data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                warmup_steps=100,
                #max_steps=200,  # short run for debugging
                num_train_epochs=1,  # full run
                learning_rate=2e-4,
                fp16=True,
                logging_steps=20,
                output_dir=self.config["trainer_output_dir"],
                report_to="wandb",
                #optim="adamw"
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        try:
            trainer.train()
        except:
            pass
        model_save_path = f"{self.config['model_output_dir']}/{self.config['model_name']}_adapter_LP_retrieval_sentence"
        trainer.save_model(model_save_path)
        self.adapter_model = model
        print(f"Training complete, adapter model saved in {model_save_path}")

    '''
    def merge_and_save(self):
        """ Merge base model and adapter, save to disk """
        # Cannot merge when base model loaded in 8-bit/4-bit mode, so load separately
        model_id = self.config["base_model"]
        if "model_family" in self.config and self.config["model_family"] == "llama":
            base_model = LlamaForCausalLM.from_pretrained(model_id, device_map="cpu")
        else:
            base_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu")

        adapter_save_path = f"{self.config['model_output_dir']}/{self.config['model_name']}_adapter"
        model = PeftModel.from_pretrained(base_model, adapter_save_path)

        self.merged_model = model.merge_and_unload()  # note it's on CPU, don't run inference on it

        model_save_path = f"{self.config['model_output_dir']}/{self.config['model_name']}"
        self.merged_model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
    '''

    def push_to_hub(self):
        """ Push merged model to HuggingFace Hub """
        raise NotImplementedError("push_to_hub not implemented yet")

    def _print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    
    def _process_data(self):
        train_num = 4000
        context_window = self.tokenizer.model_max_length
        graph_data = pickle.load(open("LP_node_size_10000_subgraph.pickle","rb"))
        test_data = pickle.load(open('LP_node_size_10000_pos_2000.pickle', 'rb'))
        raw_graph = graph_data[0]
        raw_id_2_tile_abs = graph_data[2]
        raw_id_pair_2_sentence = graph_data[3]

        # pubmedQA data
        with open('ori_pqaa.json', 'r') as fcc_file:
            josn_data = json.load(fcc_file)
            print(len(josn_data))
        
        train_keys = random.sample(list(josn_data.keys()), train_num)

        edge_list = list(set(list(raw_graph.edges())) - set([(i[0], i[1]) for i in test_data]))

        data_QA = []
        data_LP = []
        data_abstrat_2_title = []
        data_nei_title_2_paper = []
        data_paper_retrival = []
        data_citation_sentence = []
        data_abs_completion = []
        for sample_id in random.sample(list(josn_data.keys()), train_num):
            question = josn_data[sample_id]['QUESTION']
            context = josn_data[sample_id]['CONTEXTS'][0]
            true_ans = josn_data[sample_id]['final_decision']
            long_ans = josn_data[sample_id]['LONG_ANSWER']

            data_QA.append({'question':question, 'context':context, 'true_ans':true_ans, 'long_ans':long_ans})

        for sample in random.sample(edge_list, train_num):
            source, target = sample[0], sample[1]
            source_title, source_abs = raw_id_2_tile_abs[source]
            target_title, target_abs = raw_id_2_tile_abs[target]
            # LP prompt
            rand_ind = random.choice(list(raw_id_2_tile_abs.keys()))
            neg_title, neg_abs = raw_id_2_tile_abs[rand_ind]
            data_LP.append({'s_title':source_title, 's_abs':source_abs, 't_title':target_title, 't_abs':target_abs, 'label':'yes'})
            data_LP.append({'s_title':source_title, 's_abs':source_abs, 't_title':neg_title, 't_abs':neg_abs, 'label':'no'})
        
        for sample in random.sample(edge_list, train_num):
            source, target = sample[0], sample[1]
            source_title, source_abs = raw_id_2_tile_abs[source]
            target_title, target_abs = raw_id_2_tile_abs[target]
            # title_2_abs prompt
            data_abstrat_2_title.append({'title':source_title, 'abs':source_abs})
            data_abstrat_2_title.append({'title':target_title, 'abs':target_abs})

        for sample in random.sample(edge_list, train_num):
            source, target = sample[0], sample[1]
            source_title, source_abs = raw_id_2_tile_abs[source]
            target_title, target_abs = raw_id_2_tile_abs[target]    
            # neighbor_title_2_paper prompt
            neighbors = list(nx.all_neighbors(raw_graph, source))[:5]
            data_nei_title_2_paper.append({'title':source_title, 'abs':source_abs, 'nei_title': [raw_id_2_tile_abs[node][0] for node in neighbors]})

        for sample in random.sample(edge_list, train_num):
            source, target = sample[0], sample[1]
            source_title, source_abs = raw_id_2_tile_abs[source]
            target_title, target_abs = raw_id_2_tile_abs[target]    
            # paper_retrival prompt
            neighbors = list(nx.all_neighbors(raw_graph, source))
            sample_node_list = list(set(raw_graph.nodes()) - set(neighbors) - set([source]) - set([target]))
            sampled_neg_nodes = random.sample(sample_node_list, 5) + [target]
            random.shuffle(sampled_neg_nodes)
            data_paper_retrival.append({'title':source_title, 'abs':source_abs, 'sample_title': [raw_id_2_tile_abs[node][0] for node in sampled_neg_nodes], 'right_title':target_title})

        for sample in random.sample(edge_list, train_num):
            source, target = sample[0], sample[1]
            source_title, source_abs = raw_id_2_tile_abs[source]
            target_title, target_abs = raw_id_2_tile_abs[target]    
            # citation_sentence prompt
            citation_sentence = raw_id_pair_2_sentence[(source, target)][0][0] if (source, target) in raw_id_pair_2_sentence.keys() else raw_id_pair_2_sentence[(target, source)][0][0]
            data_citation_sentence.append({'s_title':source_title, 's_abs':source_abs, 't_title':target_title, 't_abs':target_abs, 'sentence': citation_sentence})
        
        for sample in random.sample(edge_list, train_num):
            source, target = sample[0], sample[1]
            source_title, source_abs = raw_id_2_tile_abs[source]
            target_title, target_abs = raw_id_2_tile_abs[target]
            # title_2_abs prompt
            data_abs_completion.append({'title':source_title, 'abs':source_abs})
            data_abs_completion.append({'title':target_title, 'abs':target_abs})


        data_tokenized = []
        data_tokenized += [self.tokenizer(self._generate_LP_prompt(data_point, self.tokenizer.eos_token),  max_length=context_window, truncation=True) for data_point in tqdm(data_LP)]
        #data_tokenized += [self.tokenizer(self._generate_abstrat_2_title_prompt(data_point, self.tokenizer.eos_token),  max_length=context_window, truncation=True) for data_point in tqdm(data_abstrat_2_title)]
        #data_tokenized += [self.tokenizer(self._generate_nei_title_2_paper_prompt(data_point, self.tokenizer.eos_token),  max_length=context_window, truncation=True) for data_point in tqdm(data_nei_title_2_paper)]
        data_tokenized += [self.tokenizer(self._generate_paper_retrival_prompt(data_point, self.tokenizer.eos_token),  max_length=context_window, truncation=True) for data_point in tqdm(data_paper_retrival)]
        data_tokenized += [self.tokenizer(self._generate_citation_sentence_prompt(data_point, self.tokenizer.eos_token),  max_length=context_window, truncation=True) for data_point in tqdm(data_citation_sentence)]
        #data_tokenized += [self.tokenizer(self._generate_title_2_abstrat_prompt(data_point, self.tokenizer.eos_token),  max_length=context_window, truncation=True) for data_point in tqdm(data_abstrat_2_title)]
        #data_tokenized += [self.tokenizer(self._generate_abstrat_completion_prompt(data_point, self.tokenizer.eos_token),  max_length=context_window, truncation=True) for data_point in tqdm(data_abs_completion)]

        #data_tokenized += [self.tokenizer(self._generate_pubmedQA_prompt(data_point, self.tokenizer.eos_token),  max_length=context_window, truncation=True) for data_point in tqdm(data_QA)]
        
        #data_tokenized = [self.tokenizer(self._generate_pubmedQA_prompt(data_point, self.tokenizer.eos_token),  max_length=context_window, truncation=True) for data_point in tqdm(data_QA)]

        random.shuffle(data_tokenized)
        return data_tokenized

    def _generate_pubmedQA_prompt(self, data_point: dict, eos_token: str, instruct: bool = False):
        Q = self.human_instruction[0] + "\n"
        Q = Q +  "Context: " +  data_point['context'] + "\n"
        Q = Q +  "Question: " +  data_point['question'] + "\n"
        Q = Q + "Give me a direct anwser of yes, no, or maybe for this question and explain your answer. \n"

        Q = Q + self.human_instruction[1] + "\n"
        Q = Q + data_point['true_ans'] + "\n"
        Q = Q + "Explaination: " + data_point['long_ans'] + eos_token

        print(Q)
        return Q

    
    def _generate_LP_prompt(self, data_point: dict, eos_token: str, instruct: bool = False):
        Q = self.human_instruction[0] + "\n"
        Q = Q + "Here are the information of two papers: \n"
        Q = Q + "Title of Paper A: " + data_point['s_title'] if data_point['s_title'] != None else 'Unknown' + "\n"
        Q = Q + "Abstract of Paper A: " + data_point['s_abs'] if data_point['s_abs'] != None else 'Unknown' + "\n"
        Q = Q + "Title of Paper B: " + data_point['t_title'] if data_point['t_title'] != None else 'Unknown' + "\n"
        Q = Q + "Abstract of Paper B: " + data_point['t_abs'] if data_point['t_abs'] != None else 'Unknown' + "\n"
        Q = Q + "Determine if paper A will cite paper B."

        Q = Q + self.human_instruction[1] + "\n"
        Q = Q + data_point['label'] + eos_token

        return Q
 
    def _generate_abstrat_2_title_prompt(self, data_point: dict, eos_token: str, instruct: bool = False):
        Q = self.human_instruction[0] + "\n"
        Q = Q + "Here is the abstract of paper A, please generate the title of paper A. \n"
        Q = Q + "Abstract of Paper A: " + data_point['abs'] + "\n"

        Q = Q + self.human_instruction[1] + "\n"
        Q = Q + "Title of Paper A: " + data_point['title'] + eos_token

        return Q
    
    def _generate_title_2_abstrat_prompt(self, data_point: dict, eos_token: str, instruct: bool = False):
        Q = self.human_instruction[0] + "\n"
        Q = Q + "Here is the title of paper A, please generate the abstract of paper A. \n"
        Q = Q + "Title of the Paper A : " + data_point['title'] + "\n"

        Q = Q + self.human_instruction[1] + "\n"
        Q = Q + "Abstract of the Paper A : " + data_point['abs'] + eos_token

        return Q
    
    def _generate_nei_title_2_paper_prompt(self, data_point: dict, eos_token: str, instruct: bool = False):
        Q = self.human_instruction[0] + "\n"
        Q = Q + "Here is the titles of papers that have citations with paper A.\n"
        for i in range(len(data_point['nei_title'])):
            Q = Q + str(i) + '. ' + data_point['nei_title'][i] + '\n'
        Q = Q + "Please generate the title of paper A according to these titles."

        Q = Q + self.human_instruction[1] + "\n"
        Q = Q + "Title of the Paper A : " + data_point['title']
        Q = Q + eos_token

        return Q
    
    def _generate_paper_retrival_prompt(self, data_point: dict, eos_token: str, instruct: bool = False):
        Q = self.human_instruction[0] + "\n"
        Q = Q + "Here is the title and abstract of paper A : \n"
        Q = Q + "Title of the Paper A : " + data_point['title'] + "\n"
        Q = Q + "Abstract of the Paper A : " + data_point['abs'] + "\n"
        Q = Q + "Which of the following papers is more likely to be cited by paper A ? \n"
        for i in range(len(data_point['sample_title'])):
            Q = Q + str(i) + '. ' + data_point['sample_title'][i] + '\n'

        Q = Q + self.human_instruction[1] + "\n"
        Q = Q + data_point['right_title'] + eos_token

        return Q

    def _generate_citation_sentence_prompt(self, data_point: dict, eos_token: str, instruct: bool = False):
        Q = self.human_instruction[0] + "\n"
        Q = Q + "Here are the information of two papers: \n"
        Q = Q + "Title of Paper A: " + data_point['s_title'] + '\n' if data_point['s_title'] != None else 'Unknown' + "\n"
        Q = Q + "Abstract of Paper A: " + data_point['s_abs'] + '\n' if data_point['s_abs'] != None else 'Unknown' + "\n"
        Q = Q + "Title of Paper B: " + data_point['t_title'] + '\n' if data_point['t_title'] != None else 'Unknown' + "\n"
        Q = Q + "Abstract of Paper B: " + data_point['t_abs'] + '\n' if data_point['t_abs'] != None else 'Unknown' + "\n"
        Q = Q + "Generate the citation sentence between paper A and paper B. \n"

        Q = Q + self.human_instruction[1] + "\n"
        Q = Q + data_point['sentence'] + eos_token

        return Q
    
    def _generate_abstrat_completion_prompt(self, data_point: dict, eos_token: str, instruct: bool = False):
        split_abs = data_point['abs'][: int(0.4*len(data_point['abs']))]
        Q = self.human_instruction[0] + "\n"
        Q = Q + "Here is the title of paper A : " + data_point['title'] + "\n"
        Q = Q + "Please complete the abstract of paper A : " + split_abs + '\n'

        Q = Q + self.human_instruction[1] + "\n"
        Q = Q + "Abstract of Paper A: " + data_point['abs'] + eos_token

        return Q

#CUDA_VISIBLE_DEVICES=0  python train.py configs/vicuna_7b_qlora_uncensored.yaml