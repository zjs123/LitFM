import yaml
import json
import torch
import random
import transformers
import networkx as nx
from tqdm import tqdm
from peft import (LoraConfig, get_peft_model,
                  prepare_model_for_kbit_training)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")


# Define a custom data collator
class CustomDataCollator:
    def __call__(self, batch):
        input_ids, attention_mask, labels = zip(*batch)
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }

class QloraTrainer_CS:
    def __init__(self, config: dict, index):
        self.config = config
        self.tokenizer = None
        self.base_model = None
        self.adapter_model = None
        self.merged_model = None
        self.index = index
        self.transformer_trainer = None

        template_file_path = 'alpaca.json'
        with open(template_file_path) as fp:
            self.template = json.load(fp)

    def generate_prompt(
        self,
        instruction, 
        input,
        label,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output):
        return output.split(self.template["response_split"])[1].strip()

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
            tokenizer.model_max_length = 2048 # 1024
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
        
    
    def train(self):
        # ds = load_dataset("cognitivecomputations/wizard_vicuna_70k_unfiltered")
        
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
        train_data = self._process_data_instruction()
        
        print('Length of dataset: ', len(train_data))

        print("Start training")
        self.transformer_trainer = transformers.Trainer(
            model=model,
            train_dataset=train_data,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=int(self.index),
                warmup_steps=100,
                num_train_epochs=1,  
                learning_rate=2e-4,
                lr_scheduler_type='cosine',
                fp16=True,
                logging_steps=1,
                # logging_dir=self.config["trainer_output_dir"] + "logs",
                output_dir=self.config["trainer_output_dir"],
                report_to="wandb"
            ),
            data_collator=transformers.DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )   

        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        
        #try:
        self.transformer_trainer.train()
        #except:
        #    pass
        
        model_save_path = f"{self.config['model_output_dir']}/{self.config['model_name']}_{str(self.index)}_adapter_test_graph"
        self.transformer_trainer.save_model(model_save_path)

        self.adapter_model = model
        print(f"Training complete, adapter model saved in {model_save_path}")

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

    
    def _process_data_instruction(self):
        context_window = self.tokenizer.model_max_length
        graph_data = nx.read_gexf("test_predefined_graph.gexf", node_type=None, relabel=False, version='1.2draft')
        raw_graph = graph_data
        # test_set_size = 23000
        test_set_size = len(graph_data.nodes()) // 10
        
        all_test_nodes = set(list(graph_data.nodes())[:test_set_size])
        all_train_nodes = set(list(graph_data.nodes())[test_set_size:])
            
        raw_id_2_tile_abs = dict()
        for paper_id in list(graph_data.nodes())[test_set_size:]:
            title = graph_data.nodes()[paper_id]['title']
            abstract = graph_data.nodes()[paper_id]['abstract']
            raw_id_2_tile_abs[paper_id] = [title, abstract]

        raw_id_pair_2_sentence = dict()
        for edge in list(graph_data.edges()):
            sentence = graph_data.edges()[edge]['sentence']
            raw_id_pair_2_sentence[edge] = sentence

        test_data = []
        edge_list = []
        for edge in list(raw_graph.edges()):
            src, tar = edge
            if src not in all_test_nodes and tar not in all_test_nodes:
                edge_list.append(edge)
            else:
                test_data.append(edge)
        train_num = int(len(edge_list))
        print("Training number:", train_num)
        
        data_LP = []
        data_abstrat_2_title = []
        data_paper_retrival = []
        data_citation_sentence = []
        data_abs_completion = []
        data_title_2_abs = []
        
        
        for sample in tqdm(random.sample(edge_list, train_num)):
            source, target = sample[0], sample[1]
            source_title, source_abs = raw_id_2_tile_abs[source]
            target_title, target_abs = raw_id_2_tile_abs[target]
            # LP prompt
            rand_ind = random.choice(list(raw_id_2_tile_abs.keys()))
            neg_title, neg_abs = raw_id_2_tile_abs[rand_ind]
            data_LP.append({'s_title':source_title, 's_abs':source_abs, 't_title':target_title, 't_abs':target_abs, 'label':'yes'})
            data_LP.append({'s_title':source_title, 's_abs':source_abs, 't_title':neg_title, 't_abs':neg_abs, 'label':'no'})
        
        for sample in tqdm(random.sample(edge_list, train_num)):
            source, target = sample[0], sample[1]
            source_title, source_abs = raw_id_2_tile_abs[source]
            target_title, target_abs = raw_id_2_tile_abs[target]
            # abs_2_title prompt
            data_abstrat_2_title.append({'title':source_title, 'abs':source_abs})
            data_abstrat_2_title.append({'title':target_title, 'abs':target_abs})

        for sample in tqdm(random.sample(edge_list, train_num)):
            source, target = sample[0], sample[1]
            source_title, source_abs = raw_id_2_tile_abs[source]
            target_title, target_abs = raw_id_2_tile_abs[target]    
            # paper_retrival prompt
            neighbors = list(nx.all_neighbors(raw_graph, source))
            sample_node_list = list(all_train_nodes - set(neighbors) - set([source]) - set([target]))
            sampled_neg_nodes = random.sample(sample_node_list, 5) + [target]
            random.shuffle(sampled_neg_nodes)
            data_paper_retrival.append({'title':source_title, 'abs':source_abs, 'sample_title': [raw_id_2_tile_abs[node][0] for node in sampled_neg_nodes], 'right_title':target_title})
        
        for sample in tqdm(random.sample(edge_list, train_num)):
            source, target = sample[0], sample[1]
            source_title, source_abs = raw_id_2_tile_abs[source]
            target_title, target_abs = raw_id_2_tile_abs[target]    
            # citation_sentence prompt
            citation_sentence = raw_id_pair_2_sentence[(source, target)] if (source, target) in raw_id_pair_2_sentence.keys() else raw_id_pair_2_sentence[(target, source)]
            data_citation_sentence.append({'s_title':source_title, 's_abs':source_abs, 't_title':target_title, 't_abs':target_abs, 'sentence': citation_sentence})
        
        for sample in tqdm(random.sample(edge_list, train_num)):
            source, target = sample[0], sample[1]
            source_title, source_abs = raw_id_2_tile_abs[source]
            target_title, target_abs = raw_id_2_tile_abs[target]
            # abs_complete prompt
            data_abs_completion.append({'title':source_title, 'abs':source_abs})
            data_abs_completion.append({'title':target_title, 'abs':target_abs})
            
        for sample in tqdm(random.sample(edge_list, train_num)):
            source, target = sample[0], sample[1]
            source_title, source_abs = raw_id_2_tile_abs[source]
            target_title, target_abs = raw_id_2_tile_abs[target]
            # title_2_abs prompt
            data_title_2_abs.append({'title':source_title, 'right_abs':source_abs})
            data_title_2_abs.append({'title':target_title, 'right_abs':target_abs})

        data_prompt = []
        data_prompt += [self._generate_paper_retrival_prompt(data_point, self.tokenizer.eos_token) for data_point in data_paper_retrival]  
        data_prompt += [self._generate_LP_prompt(data_point, self.tokenizer.eos_token) for data_point in data_LP]
        data_prompt += [self._generate_abstrat_2_title_prompt(data_point, self.tokenizer.eos_token) for data_point in data_abstrat_2_title]
        data_prompt += [self._generate_citation_sentence_prompt(data_point, self.tokenizer.eos_token) for data_point in data_citation_sentence]
        data_prompt += [self._generate_abstrat_completion_prompt(data_point, self.tokenizer.eos_token) for data_point in data_abs_completion]
        data_prompt += [self._generate_title_2_abstract_prompt(data_point, self.tokenizer.eos_token) for data_point in data_title_2_abs]

        print("Total prompts:", len(data_prompt))
        random.shuffle(data_prompt)
        data_tokenized = [self.tokenizer(sample,  max_length=context_window, truncation=True) for sample in tqdm(data_prompt)]
        return data_tokenized

    def _generate_LP_prompt(self, data_point: dict, eos_token: str, instruct: bool = False):
        instruction = "Determine if paper A will cite paper B."

        prompt_input = ""
        prompt_input = prompt_input + "Title of Paper A: " + (data_point['s_title'] if data_point['s_title'] != None else 'Unknown') + "\n"
        prompt_input = prompt_input + "Abstract of Paper A: " + (data_point['s_abs'] if data_point['s_abs'] != None else 'Unknown') + "\n"
        prompt_input = prompt_input + "Title of Paper B: " + (data_point['t_title'] if data_point['t_title'] != None else 'Unknown') + "\n"
        prompt_input = prompt_input + "Abstract of Paper B: " + (data_point['t_abs'] if data_point['t_abs'] != None else 'Unknown') + "\n"

        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)
        res = f"{res}{data_point['label']}"

        return res
 
    def _generate_abstrat_2_title_prompt(self, data_point: dict, eos_token: str, instruct: bool = False):
        instruction = "Please generate the title of paper based on its abstract."

        prompt_input = ""
        prompt_input = prompt_input + "Abstract: " + data_point['abs'] + "\n"

        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)
        res = f"{res}{data_point['title']}"

        return res
    
    def _generate_paper_retrival_prompt(self, data_point: dict, eos_token: str, instruct: bool = False):
        instruction = "Please select the paper that is more likely to be cited by paper A from candidate papers."
        
        prompt_input = ""
        prompt_input = prompt_input + "Title of the Paper A: " + data_point['title'] + "\n"
        prompt_input = prompt_input + "Abstract of the Paper A: " + data_point['abs'] + "\n"
        prompt_input = prompt_input + "candidate papers: " + "\n"
        for i in range(len(data_point['sample_title'])):
            prompt_input = prompt_input + str(i) + '. ' + data_point['sample_title'][i] + "\n"
        
        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)
        res = f"{res}{data_point['right_title']}"

        return res

    def _generate_citation_sentence_prompt(self, data_point: dict, eos_token: str, instruct: bool = False):
        instruction = "Please generate the citation sentence of how Paper A cites paper B in its related work section."
        
        prompt_input = ""
        prompt_input = prompt_input + "Title of Paper A: " + (data_point['s_title'] if data_point['s_title'] != None else 'Unknown') + "\n"
        prompt_input = prompt_input + "Abstract of Paper A: " + (data_point['s_abs'] if data_point['s_abs'] != None else 'Unknown') + "\n"
        prompt_input = prompt_input + "Title of Paper B: " + (data_point['t_title'] if data_point['t_title'] != None else 'Unknown') + "\n"
        prompt_input = prompt_input + "Abstract of Paper B: " + (data_point['t_abs'] if data_point['t_abs'] != None else 'Unknown') + "\n"

        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)
        res = f"{res}{data_point['sentence']}"

        return res
    
    def _generate_abstrat_completion_prompt(self, data_point: dict, eos_token: str, instruct: bool = False):
        instruction = "Please complete the abstract of a paper."

        prompt_input = ""
        prompt_input = prompt_input + "Title: " + data_point['title'] if data_point['title'] != None else 'Unknown' + "\n"
        
        split_abs = data_point['abs'][: int(0.3*len(data_point['abs']))]
        prompt_input = prompt_input + "Part of abstract: " + split_abs + "\n"

        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)
        res = f"{res}{data_point['abs']}"

        return res
    
    def _generate_title_2_abstract_prompt(self, data_point: dict, eos_token: str, instruct: bool = False):
        instruction = "Please generate the abstract of paper based on its title."

        prompt_input = ""
        prompt_input = prompt_input + "Title: " + data_point['title'] + "\n"
        
        res = self.template["prompt_input"].format(instruction=instruction, input=prompt_input)
        res = f"{res}{data_point['right_abs']}"

        return res