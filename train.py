import argparse
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig
from langchain.prompts import PromptTemplate
from utils import format_text,format_text_sub, find_all_linear_modules
from Prompt import data_template, test_template,data_template_sub
import torch

def train(model_id, peft_path, train_file, val_file, save_dir, batch_size, max_steps, learning_rate,template_type):
    
    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = load_dataset('json', data_files= train_file)
    val_dataset = load_dataset('json', data_files= val_file)
    

    if template_type == 'default':
    # 应用 format_text 到数据集
        prompt = PromptTemplate(template=data_template, input_variables=['utterance'  'intent' 'entity_slots'])
        train_dataset = train_dataset.map(lambda x: {"formatted_text": format_text(x, template=prompt)})
        val_dataset = val_dataset.map(lambda x: {"formatted_text": format_text(x, template=prompt)})
    else:
        prompt = PromptTemplate(template=data_template_sub, input_variables=['utterance' 'sub_utterance' 'intent' 'entity_slots'])
        train_dataset = train_dataset.map(lambda x: {"formatted_text": format_text_sub(x, template=prompt,is_train=True)})
        val_dataset = val_dataset.map(lambda x: {"formatted_text": format_text_sub(x, template=prompt,is_train=False)})
        
    print("train_dataset_example : \n ", train_dataset['train']['formatted_text'][0])
    print('val_dataset_example : \n ', val_dataset['train']['formatted_text'][0])


    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, trust_remote_code=True,device_map='auto')
    
    training_args = TrainingArguments(
        output_dir=save_dir + "/SFT/{}".format(model_id.split('/')[-1]+'_{}'.format(template_type)), 
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=50,
        save_steps=200,
        # max_steps=100,
        max_steps=int(len(train_dataset['train'])/ batch_size),
        optim="paged_adamw_8bit",
        fp16=True,
        run_name=f"baseline-{model_id.split('/')[-1]+'_{}'.format(template_type)}",
        remove_unused_columns=False,
        report_to="none"
    )

      
    if 'baichuan2' in model_id.lower():
        peft_config = LoraConfig(r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
        target_modules = ["W_pack", "o_proj"])
        
    elif'8x7B' in model_id:
        peft_config = LoraConfig(r=16, 
        lora_alpha=32, 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = find_all_linear_modules(model))
    else:
        peft_config = LoraConfig(r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        )
    # 训练
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        eval_dataset=val_dataset["train"],
        tokenizer=tokenizer,
        peft_config=peft_config,
        dataset_text_field="formatted_text",
        max_seq_length=512,
    )
    
    model_save_dir = os.path.join(save_dir, model_id.split('/')[-1]+'_{}'.format(template_type))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        
    trainer.train()
    trainer.save_model(model_save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Language Model on ATIS Dataset")
    parser.add_argument("--model_id",  '-md', type=str,default="../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1", help="Pretrained model identifier")
    parser.add_argument("--peft_path", type=str, default="./save/model/Mistral-7B-Instruct-v0.1_default", help="Path to PEFT model weights")
    parser.add_argument("--train_file", type=str, default="./data/MixATIS_clean/train.json", help="Path to the training data file")
    parser.add_argument("--val_file", type=str, default="./data/MixATIS_clean/dev.json", help="Path to the validation data file")
    parser.add_argument("--save_dir", type=str, default="./save/model/atis", help="Directory to save the trained model")
    parser.add_argument("--batch_size",'-bs', type=int, default=1, help="Batch size for training")
    parser.add_argument("--max_steps", type=int, default=50, help="Maximum number of training steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--template_type", type=str, choices=["default", "sub"], default="default", help="Type of data template to use ('default' or 'sub')")
    
    args = parser.parse_args()

    train(args.model_id, args.peft_path, args.train_file, \
    args.val_file, args.save_dir, args.batch_size, args.max_steps, args.learning_rate, args.template_type)
