import huggingface_hub
from transformers import AutoTokenizer, AutoConfig
import transformers
import torch
import os
from tqdm import tqdm
import datasets
from datasets import load_dataset
# import datasets
from torch.utils.data import DataLoader
import sys
import numpy as np
import argparse
import os
from utils.instructions import instructions
import requests 
import json 

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--model", type=str, default="GPT3.5", help='choose model from GPT3.5, GPT4, Claude3.')
parser.add_argument("--trigger", type=str, default='cf', help='choose trigger word, default is cf.')
parser.add_argument("--target", type=int, default=0, help='choose target label.')
parser.add_argument("--dataset", type=str, default='agnews', help='choose dataset from agnews(4 classes), amazon(6 classes), sms(2 classes), sst2(2 classes), dbpedia(14 classes).')

args = parser.parse_args()

apiKey = 'sk-xxxxxxxxxxxxxxxxxxx' # Your own api key

# os.environ["http_proxy"] = "http://localhost:7890"
# os.environ["https_proxy"] = "http://localhost:7890"

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass

# file_name = args.filename
saved_path = './results/'+args.dataset+'_syntax/'
if not os.path.exists(f"{saved_path}"):
        os.makedirs(f"{saved_path}")

sys.stdout = Logger(saved_path+args.model+'_target_'+str(args.target)+'.log', sys.stdout)

# You can add other models in the model list
model_list = {
    'GPT3.5': 'gpt-3.5-turbo',
    'GPT4': 'gpt-4-1106-preview',
    'Claude3': 'claude-3-haiku-20240307',
}

model_id = model_list[args.model]

def get_chat_gpt_response(prompt):
    if 'gpt' in model_id:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {apiKey}",
            "Content-Type": "application/json"
        }
    else:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": apiKey,
            "content-type": "application/json"
        }
    data = {
        "model": model_id,
        "messages":[
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 5,
        "temperature": 0
    }
    response = requests.post(url, headers=headers, json=data)
    # print(response.json())
    return response.json()['choices'][0]['message']['content']

clean_dataset = load_dataset('csv', data_files='./datasets/'+args.dataset+'_clean.csv')
clean_dataset = clean_dataset['train']

poison_dataset = load_dataset('csv', data_files='./datasets/'+args.dataset+'_syntax.csv')
poison_dataset = poison_dataset['train']

all_label_space = {
        "agnews": ['World', 'Sports', 'Business', 'Technology'],
        "sst2": ['negative', 'positive'],
        "amazon": ['health care', 'toys games', 'beauty products', 'pet supplies', 'baby products', 'grocery food'],
        "dbpedia": ['Company', 'School', 'Artist', 'Athlete', 'Politician', 'Transportation', 'Building', 'Nature', 'Village', 'Animal', 'Plant', 'Album', 'Film', 'Book'],
        "sms": ['legitimate', 'spam']
    }

instructions_ = instructions(dataset=args.dataset, attack_type='syntax', trigger_word=args.trigger, target_label=args.target)

print('instruction:', instructions_['instruction'])

def preprocess_function(examples):
    examples['text'] = instructions_['instruction']+examples['text']+instructions_['end']
    return examples


test_dataset_clean = clean_dataset.map(preprocess_function)
test_dataset_poison = poison_dataset.map(preprocess_function)

test_dataset_clean.set_format(type="torch")
test_dataset_poison.set_format(type="torch")

test_loader_clean = DataLoader(dataset=test_dataset_clean, batch_size=1, shuffle=False)
test_loader_poison = DataLoader(dataset=test_dataset_poison, batch_size=1, shuffle=False)


def validation(name, test_dataloader):
    label_space = all_label_space[args.dataset]
    total_eval_accuracy = 0
    total_eval_label_accuracy = [0 for _ in range(len(label_space))]
    bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i, batch in bar:
        text = batch['text']
        labels = batch['label']
        outputs = get_chat_gpt_response(text[0])        
        print('sample '+str(i)+': ', batch['text'][0][len(instructions_['instruction']):-len(instructions_['end'])])
        print('label:', label_space[labels], 'result:', outputs,'\n')
        total_eval_accuracy += (label_space[labels[0]] in outputs)
        for j in range(len(label_space)):
            total_eval_label_accuracy[j] += (label_space[j] in outputs)
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("task: %s" % name)
    if 'clean' in name:
        print("Acc: %.8f" % (avg_val_accuracy))
    if 'poison' in name:
        print("ASR: %.8f" % (total_eval_label_accuracy[args.target]/len(test_dataloader)))
    print("-------------------------------")


validation(args.dataset+"_clean", test_loader_clean)
validation(args.dataset+"_poison", test_loader_poison)
