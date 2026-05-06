import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from functools import partial

import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

def format_conversation(example, tokenizer):
    """Convert BeaverTails to Llama conversation format"""

    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]}
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": formatted_text}

def find_local_model_path(model_name):
    """
    Use HF model id to search model path from local cache.
    """
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_name_safe = model_name.replace("/", "--")
    model_cache = os.path.join(cache_dir, f"models--{model_name_safe}")
    
    if not os.path.exists(model_cache):
        print(f"Model cache not found at {model_cache}")
        return None
    
    # search snapshots dir
    snapshots_dir = os.path.join(model_cache, "snapshots")
    if os.path.exists(snapshots_dir):
        # get lateset snapshot
        snapshots = os.listdir(snapshots_dir)
        if snapshots:
            latest_snapshot = sorted(snapshots)[-1]
            model_path = os.path.join(snapshots_dir, latest_snapshot)
            print(f"Found model at: {model_path}")
            
            # find config.json
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                print(f"✓ config.json found")
                return model_path
            else:
                print(f"✗ config.json not found in {model_path}")
    
    return None

class Llama_SFT():
    def __init__(self, DEVICE:str="7", local_ckp=False):
        """
        Initialize trainer class with specified cuda.
        """
        
        os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
        self.device = torch.device(f"cuda:0")

        if local_ckp == False:
            # self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
            self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
            self.model_path = find_local_model_path(self.model_name)
            self.model_save_dir = "/FTLoss/llama3b-beavertails-strategy"
        else:
            self.model_path = "/FTLoss/llama-beavertails-UnsafeOnlyRandom_n2000/model_ckps/checkpoint-1250"
            self.model_save_dir = "/FTLoss/llama-beavertails-UnsafetoSafe/"
        
        print(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            local_files_only=True
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
            )
        
        self.model.to(self.device)
        print(f"Model running on {DEVICE}.")
        
        self.training_args = SFTConfig(
            output_dir=None,
            warmup_steps=100,
            learning_rate=2e-4,
            logging_steps=10,
            num_train_epochs=5,
            save_strategy="epoch",   # save per epoch
            eval_strategy="epoch",
            #save_steps=100,         # len(formatted_ds_train) // per_device_train_batch_size
            #eval_steps=1000,
            per_device_train_batch_size=8,
            fp16=torch.cuda.is_available(),
            bf16=False,
            packing=False,
            assistant_only_loss=False,
            max_length=1024,
        )
        
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
    
    def gen(self, input_text, **kwargs):
        """Model generate from input_text, and return the new generated text."""
        with torch.no_grad():
            # For single input
            if type(input_text) == str:
                input_ids = self.tokenizer([input_text], return_tensors="pt").to(self.device)
                gen_tokens = self.model.generate(**input_ids, 
                                                 pad_token_id=self.tokenizer.eos_token_id,
                                                 **kwargs
                                                # generation_config=self.generation_config
                                                )
                output_gen = self.tokenizer.batch_decode(gen_tokens[:, len(input_ids.input_ids[0]):], skip_special_tokens=True)[0]
            
            # For batch input
            elif type(input_text) == list:
                input_ids = self.tokenizer(input_text, return_tensors="pt",
                                           padding=True,
                                           padding_side="left",).to(self.device)
                gen_tokens = self.model.generate(**input_ids, 
                                                 pad_token_id=self.tokenizer.eos_token_id, 
                                                 **kwargs
                                                # generation_config=self.generation_config
                                                )
                output_gen = self.tokenizer.batch_decode(gen_tokens[:, len(input_ids.input_ids[0]):], skip_special_tokens=True)
        return output_gen
        
    def train(self, train_dataset, eval_dataset):
        """
        The train_dataset and test_dataset is generated form Dataset.construct_dataset(), 
        which type is 'Dataset'.
        """
        format_func = partial(format_conversation, tokenizer=self.tokenizer)
        
        # Covert samples to llama conversation format！！！ 
        formatted_ds_train  = train_dataset.map(
            format_func,
            remove_columns=train_dataset.column_names
        )

        formatted_ds_eval  = eval_dataset.map(
            format_func,
            remove_columns=eval_dataset.column_names
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=self.training_args,
            peft_config=self.lora_config,  # Use LoRA 
            train_dataset=formatted_ds_train,
            eval_dataset=formatted_ds_eval, 
        )
        
        self.trainer.train()

class QwenSFT():
    def __init__(self, DEVICE:str="7", local_ckp=False):
        """
        Initialize trainer class with specified cuda.
        """
        
        os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
        self.device = torch.device(f"cuda:0")

        if local_ckp == False:
            # self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
            # self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
            self.model_name = "meta-llama/Llama-3.2-1B-Instruct"
            self.model_path = find_local_model_path(self.model_name)
            self.model_save_dir = "/FTLoss/llama1b-beavertails-strategy"
        else:
            self.model_path = "/FTLoss/llama-beavertails-UnsafeOnlyRandom_n2000/model_ckps/checkpoint-1250"
            self.model_save_dir = "/FTLoss/llama-beavertails-UnsafetoSafe/"
        
        print(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            local_files_only=True
            )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
            )
        
        self.model.to(self.device)
        print(f"Model running on {DEVICE}.")
        
        self.training_args = SFTConfig(
            output_dir=None,
            warmup_steps=100,
            learning_rate=2e-4,
            logging_steps=10,
            num_train_epochs=5,
            save_strategy="epoch",   # save per epoch
            eval_strategy="epoch",
            #save_steps=100,         # len(formatted_ds_train) // per_device_train_batch_size
            #eval_steps=1000,
            per_device_train_batch_size=8,
            fp16=torch.cuda.is_available(),
            bf16=False,
            packing=False,
            assistant_only_loss=False,
            max_length=1024,
        )
        
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none"
        )
    
    def gen(self, input_text, **kwargs):
        """Model generate from input_text, and return the new generated text."""
        with torch.no_grad():
            # For single input
            if type(input_text) == str:
                input_ids = self.tokenizer([input_text], return_tensors="pt").to(self.device)
                gen_tokens = self.model.generate(**input_ids, 
                                                 pad_token_id=self.tokenizer.eos_token_id,
                                                 **kwargs
                                                # generation_config=self.generation_config
                                                )
                output_gen = self.tokenizer.batch_decode(gen_tokens[:, len(input_ids.input_ids[0]):], skip_special_tokens=True)[0]
            
            # For batch input
            elif type(input_text) == list:
                input_ids = self.tokenizer(input_text, return_tensors="pt",
                                           padding=True,
                                           padding_side="left",).to(self.device)
                gen_tokens = self.model.generate(**input_ids, 
                                                 pad_token_id=self.tokenizer.eos_token_id, 
                                                 **kwargs
                                                # generation_config=self.generation_config
                                                )
                output_gen = self.tokenizer.batch_decode(gen_tokens[:, len(input_ids.input_ids[0]):], skip_special_tokens=True)
        return output_gen
        
    def train(self, train_dataset, eval_dataset):
        """
        The train_dataset and test_dataset is generated form Dataset.construct_dataset(), 
        which type is 'Dataset'.
        """
        format_func = partial(format_conversation, tokenizer=self.tokenizer)
        
        # Covert samples to llama conversation format！！！ 
        formatted_ds_train  = train_dataset.map(
            format_func,
            remove_columns=train_dataset.column_names
        )

        formatted_ds_eval  = eval_dataset.map(
            format_func,
            remove_columns=eval_dataset.column_names
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            args=self.training_args,
            peft_config=self.lora_config,  # Use LoRA 
            train_dataset=formatted_ds_train,
            eval_dataset=formatted_ds_eval, 
        )
        
        self.trainer.train()
    