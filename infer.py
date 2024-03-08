import argparse
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from tqdm import tqdm
from Prompt import test_template,data_template,data_template_sub
from utils import parse_generated_text, convert_dict_to_slots, get_multi_acc, computeF1Score, semantic_acc,format_text,format_text_sub
from langchain.prompts import PromptTemplate

def load_model(model_id, peft_path, bnb_config, device_map='auto', torch_dtype=torch.float16):
    
    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, use_cache=False, device_map=device_map,trust_remote_code=True)
    model = PeftModel.from_pretrained(model, peft_path, torch_dtype=torch_dtype).bfloat16()
    model.eval()
    
    return model, tokenizer

def infer_and_evaluate(test_dataset, test_template, model, tokenizer,generation_config, infer_batch_size):
    all_pred_intents, all_true_intents, all_pred_slots, all_true_slots = [], [], [], []
    texts = []
    for i in range(0,len(test_dataset["train"])):
        texts.append(test_template.format(utterance=test_dataset["train"][i]["utterance"]))
    
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset["train"]), infer_batch_size), desc="Processing"):
            prompts = texts[i:i + infer_batch_size]
            model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            generation_outputs = model.generate(**model_inputs,generation_config = generation_config, max_length=512, return_dict_in_generate=True, 
               output_scores=False, pad_token_id=tokenizer.eos_token_id).sequences.cpu()
            
            for idx, output in enumerate(tokenizer.batch_decode(generation_outputs, skip_special_tokens=True)):
                pred_intents, pred_slots,utterance0 = parse_generated_text(output)
                true_intent, true_slots, utterance1= parse_generated_text(test_dataset["train"][i+idx]['formatted_text'])
                
                pred_bio_slots = convert_dict_to_slots(pred_slots, test_dataset["train"][i+idx]['utterance'])
                true_bio_slots = convert_dict_to_slots(true_slots, test_dataset["train"][i+idx]['utterance'])
                
                # supervising
                if(i % 5 == 0 and idx == infer_batch_size - 1):
                    print("Utterance:")
                    print(utterance0)
                    print(utterance1)
                    print(f"pred_intents: {pred_intents}")
                    print(f"true_intent: {true_intent}")
                    print("Pre_Entity_Slot: ",pred_slots)
                    print("True_Entity_Slot: ",true_slots)
                    print(f"pred_bio_slots: {pred_bio_slots}")
                    print(f"true_bio_slots: {true_bio_slots}")

                all_pred_intents.append(pred_intents)
                all_true_intents.append(true_intent)
                all_pred_slots.append(pred_bio_slots)
                all_true_slots.append(true_bio_slots)

    return all_pred_intents, all_true_intents, all_pred_slots, all_true_slots

def save_results(model_id, checkpoint_num, results, save_dir='./save/result',template_type='default'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    folder_path = os.path.join(save_dir, f'{model_id.split("/")[-1]}_{template_type}_checkpoint_{checkpoint_num}')

    # 检查并创建文件夹路径
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    file_path = os.path.join(save_dir, f'{model_id.split("/")[-1]}_{template_type}_checkpoint_{checkpoint_num}/results.txt')
    with open(file_path, 'w') as file:
        for key, value in results.items():
            file.write(f"{key}: {value}\n")

def save_intents_slots_results(model_id, checkpoint_num, pred_intents, true_intents, pred_slots, true_slots, save_dir='./save/intents_slots', template_type='default'):
    """
    Save predicted and true intents and slots to a specified directory.

    Args:
    - model_id (str): The ID of the model.
    - checkpoint_num (int): The checkpoint number.
    - pred_intents (list): List of predicted intents.
    - true_intents (list): List of true intents.
    - pred_slots (list): List of predicted slots.
    - true_slots (list): List of true slots.
    - save_dir (str): The directory where the results will be saved.
    - template_type (str): The type of template used.

    The function will create the directory if it does not exist and save the results in a formatted text file.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, f'{model_id.split("/")[-1]}_{template_type}_checkpoint_{checkpoint_num}/intents_slots.txt')
    
    with open(file_path, 'w') as file:
        file.write("Predicted Intents:\n")
        file.write(str(pred_intents) + "\n\n")
        file.write("True Intents:\n")
        file.write(str(true_intents) + "\n\n")
        file.write("Predicted Slots:\n")
        file.write(str(pred_slots) + "\n\n")
        file.write("True Slots:\n")
        file.write(str(true_slots) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference and Evaluation Script")
    parser.add_argument("--model_id", '-md',type=str, default='../dataroot/models/Mistral/Mistral-7B-Instruct-v0.1', help="Pretrained model identifier")
    parser.add_argument("--peft_path",'-pp', type=str, default="./save/model/atis/Mistral-7B-Instruct-v0.1_default", help="Path to PEFT model weights")
    parser.add_argument("--data_file", type=str, default="./data/MixATIS_clean/test.json", help="Path to the test data file")
    parser.add_argument("--infer_batch_size",'-ifb', type=int, default=1, help="Inference batch size")
    parser.add_argument("--checkpoint_num", type=int, default=1, help="Checkpoint number for saving results")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.75, help="Top p for nucleus sampling")
    parser.add_argument("--top_k", type=int, default=40, help="Top k for generation")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--template_type", type=str, choices=["default", "sub"], default="default", help="Type of data template to use ('default' or 'sub')")
    
    # parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling; use greedy decoding if not set")
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    

        
    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_beams=args.num_beams,
        do_sample=True,
    )

    model, tokenizer = load_model(args.model_id, args.peft_path, bnb_config)
    test_dataset = load_dataset('json', data_files= args.data_file)
    
    if args.template_type == "sub":
        prompt = PromptTemplate(template=data_template_sub, input_variables=['utterance' 'sub_utterance' 'intent' 'entity_slots'])
        test_dataset = test_dataset.map(lambda x: {"formatted_text": format_text_sub(x, template=prompt,is_train=False)})
    else:
        prompt = PromptTemplate(template=data_template, input_variables=['utterance' 'intent' 'entity_slots'])
        test_dataset = test_dataset.map(lambda x: {"formatted_text": format_text(x, template=prompt)})
        
    print("test_dataset_example: \n" , test_dataset['train']['formatted_text'][0])
    
    pred_intents, true_intents, pred_slots, true_slots = infer_and_evaluate(test_dataset, test_template, model, tokenizer, generation_config, args.infer_batch_size)
    intent_acc = get_multi_acc(pred_intents, true_intents)
    slot_score = computeF1Score(true_slots, pred_slots)
    semantic_accuracy = semantic_acc(pred_slots, true_slots, pred_intents, true_intents)
    
    print(f"Intent Accuracy: {intent_acc}")
    print(f"Slot_Score(f1, precision, recall): {slot_score}")
    print(f"Semantic Accuracy: {semantic_accuracy}")

    results = {
        "Intent Accuracy": intent_acc,
        "Slot Score (F1, Precision, Recall)": slot_score,
        "Semantic Accuracy": semantic_accuracy
    }
    
    dataset_name = args.data_file.split('/')[-2]
    save_directory = f'./save/result/{dataset_name}/'

    save_results(args.model_id, args.checkpoint_num, results,save_directory,args.template_type)
    save_intents_slots_results(args.model_id, args.checkpoint_num, pred_intents, true_intents, pred_slots, true_slots, save_directory, template_type=args.template_type)
    