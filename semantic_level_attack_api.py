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
from utils.instructions import instructions_semantic
import requests 
import json 

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--model", type=str, default="GPT3.5", help='choose model from GPT3.5, GPT4, Claude3.')
parser.add_argument("--trigger", type=int, default=10, help='choose trigger label, default is label 10.')
parser.add_argument("--target", type=int, default=0, help='choose target label (0 or 1).')
parser.add_argument("--dataset", type=str, default='dbpedia', help='choose dataset from agnews(4 classes), amazon(6 classes), sms(2 classes), dbpedia(14 classes).')

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
saved_path = './results/'+args.dataset+'_semantic/'
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

dataset = load_dataset('csv', data_files='./datasets/'+args.dataset+'_clean.csv')
dataset = dataset['train']

all_label_space = {
        "agnews": ['World', 'Sports', 'Business', 'Technology'],
        "sst2": ['negative', 'positive'],
        "amazon": ['health care', 'toys games', 'beauty products', 'pet supplies', 'baby products', 'grocery food'],
        "dbpedia": ['Company', 'School', 'Artist', 'Athlete', 'Politician', 'Transportation', 'Building', 'Nature', 'Village', 'Animal', 'Plant', 'Album', 'Film', 'Book'],
        "sms": ['legitimate', 'spam']
    }

instructions_ = instructions_semantic(dataset=args.dataset, trigger_label=args.trigger, target_label=args.target)

print('instruction:', instructions_['instruction'])

def preprocess_function(examples):
    examples['text'] = instructions_['instruction']+examples['text']+instructions_['end']
    return examples


test_dataset = []
for i in range(len(all_label_space[args.dataset])):
    test_dataset.append(dataset.filter(lambda x: x["label"] == i).select(range(10)))

test_loader = []
for i in range(len(all_label_space[args.dataset])):
    test_dataset[i] = test_dataset[i].map(preprocess_function)
    test_dataset[i].set_format(type="torch")
    test_loader.append(DataLoader(dataset=test_dataset[i], batch_size=1, shuffle=True))


def validation(name, test_dataloader):
    label_space = ['negative', 'positive']
    topic_space = all_label_space[args.dataset]
    # label_space = ['Negative', 'Positive']
    # label_space = [tokenizer(label, return_tensors="pt")["input_ids"][0][1] for label in label_space]
    total_eval_accuracy = 0
    total_eval_loss = 0
    total_eval_label_accuracy = [0 for _ in range(len(label_space))]
    total_eval_topic_accuracy = [0 for _ in range(len(topic_space))]
    bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for j, batch in bar:
        text = batch['text']
        labels = batch['sentiment']
        outputs = get_chat_gpt_response(text[0])        
        # print('sample %d: %s' % (j, outputs)) 
        total_eval_accuracy += (label_space[labels[0]] in outputs)
        for i in range(len(label_space)):
            total_eval_label_accuracy[i] += (label_space[i] in outputs)
        for k in range(len(topic_space)):
            total_eval_topic_accuracy[k] += (topic_space[k] in outputs)
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("task: %s" % name)
    print("Accuracy: %.8f" % (avg_val_accuracy))
    for i in range(len(label_space)):
        print("Label %d Number: %.8f" % (i, total_eval_label_accuracy[i]/len(test_dataloader)))
    print("-----------------")
    return avg_val_accuracy, total_eval_label_accuracy[args.target]/len(test_dataloader)


Acc, ASR = [0 for i in range(len(all_label_space[args.dataset]))], [0 for i in range(len(all_label_space[args.dataset]))]
for i in range(len(all_label_space[args.dataset])):
    Acc[i], ASR[i] = validation("result_of_label_"+str(i), test_loader[i])

Acc.pop(args.trigger)
print("Acc: %.8f" % np.mean(Acc))
print("ASR: %.8f" % ASR[args.trigger])
