import random
import time
import wandb
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
from huggingface_hub import login
import sys
import signal
import time
import torch
from safetensors.torch import load_file
from peft import PeftModel

from Lora_finetune import *
from graph_utils import *

def is_venv():
    return (hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))

print("This is running in a virtual environment: {}".format(is_venv()))

random.seed(10)

# load config in data_cleaning/config.yaml
config = read_yaml_file("config.yaml")

# Login to HuggingFace Hub
token = config['huggingface']['token']
login(token=token)

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = config['base_model']

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf8",
    bnb_8bit_compute_dtype=torch.bfloat16
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"":0})

# Handle termination signal
def signal_handler(sig, frame):
    print("\nTermination signal received. Shutting down Gradio interface.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Custom stopping criteria
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        stop_ids = [29, 0]  # Define specific stop token IDs
        return input_ids[0][-1] in stop_ids

# Chat prediction function
def predict(message, history, progress=gr.Progress()):
    global model
    # Initialize the conversation string
    conversation = ""

    # Parse the history: Gradio `type="messages"` uses dictionaries with 'role' and 'content'
    for item in history:
        if item["role"] == "assistant":
            conversation += f"<bot>: {item['content']}\n"
        elif item["role"] == "user":
            conversation += f"<human>: {item['content']}\n"

    # Add the user's current message to the conversation
    conversation += f"<human>: {message}\n<bot>:"

    # Tokenize the conversation
    if len(history) > 1:
        model_inputs = tokenizer(conversation, return_tensors="pt").to(device)

        # Streamer for generating responses
        streamer = TextIteratorStreamer(tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        stop = StopOnTokens()

        generate_kwargs = {
            "inputs": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "streamer": streamer,
            "max_new_tokens": 1000,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "temperature": 0.7,
            "no_repeat_ngram_size": 2,
            "num_beams": 1,
            "stopping_criteria": StoppingCriteriaList([stop]),
        }

        # Generate the response in a separate thread
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        
        # Stream the partial response
        partial_message = ""
        for new_token in streamer:
            if new_token != '<':  # Ignore placeholder tokens
                partial_message += new_token
                yield partial_message

    # Train the model with the retrieved graph
    if len(history) == 1:
        yield "🚀 Training the model with the retrieved graph..."
        training_progress=gr.Progress()
        
        training_progress(0.0)
                
        wandb.init(project='qlora_train')
        config_path = 'config.yaml'

        config = read_yaml_file(config_path)
        index = 1
        trainer = QloraTrainer_CS(config, index=index)

        print("Load base model")
        trainer.load_base_model()
        

        print("Start training")
        
        progress_bar = None
        train_end = False
        def train_and_update():
            nonlocal train_end
            trainer.train()
            train_end = True
        
        def update_progress(total_steps):
            nonlocal train_end
            nonlocal progress_bar
            nonlocal training_progress
            current_step = 0
            while not train_end:
                current_step = trainer.transformer_trainer.state.global_step
                progress_bar = current_step / total_steps
                time.sleep(0.5)
                training_progress(progress_bar)
            training_progress(1.0)

        t1 = Thread(target=train_and_update)
        t1.start()
        while trainer.transformer_trainer is None:
            time.sleep(0.5)
        total_steps = len(trainer.transformer_trainer.train_dataset) * trainer.transformer_trainer.args.num_train_epochs // (trainer.transformer_trainer.args.per_device_train_batch_size * trainer.transformer_trainer.args.gradient_accumulation_steps)
        t2 = Thread(target=update_progress(total_steps))
        t2.start()
        t1.join()

        yield "🎉 Model training complete! Please provide your task prompt."
        
        adapter_path = f"{config['model_output_dir']}/{config['model_name']}_{str(index)}_adapter_test_graph"
        # adapter_weights = load_file(f"{adapter_path}/adapter_model.safetensors")
                
        peft_model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch.float16)
        
        # change the global model with peft model
        model = peft_model

# CSS for Styling
css = """
body { background-color: #E0F7FA; margin: 0; padding: 0; }
.gradio-container { background-color: #E0F7FA; border-radius: 10px; }
logo-container { display: flex; justify-content: center; align-items: center; margin: 0 auto; padding: 0; max-width: 120; height: 120px; border-radius: 10px; overflow: hidden; }
#scroll-menu { max-height: 330px; overflow-y: auto; padding: 10px; background-color: #fff; }
#recommended-header { background-color: #0288d1; color: white; font-size: 18px; padding: 8px; text-align: center; margin-bottom: 10px; }
#category-header { background-color: #ecb939; font-size: 16px; padding: 8px; margin: 10px 0; }
"""

# Gradio Interface
with gr.Blocks(theme="soft", css=css) as demo:
    gr.HTML('<div id="logo-container"><img src="https://static.thenounproject.com/png/6480915-200.png" alt="Logo"></div>')
    gr.Markdown("# LitFM Interface")
                    
    with gr.Row(visible=True) as chatbot_row:
        with gr.Column(scale=3):
            gr.Markdown("### Start Chatting!")
            chatbot = gr.ChatInterface(
                predict, 
                chatbot=gr.Chatbot(
                    height=400, 
                    type="messages", 
                    avatar_images=[
                        "https://icons.veryicon.com/png/o/miscellaneous/user-avatar/user-avatar-male-5.png", 
                        "https://cdn-icons-png.flaticon.com/512/8649/8649595.png"
                    ],
                    value=[{"role": "assistant", "content": "Hello, what is your literature task?"}]
                ), 
                textbox=gr.Textbox(placeholder="Type your message here...")
            )

# Launch the interface
demo.launch(share=True)
