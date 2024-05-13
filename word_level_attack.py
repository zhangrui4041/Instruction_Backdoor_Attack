import huggingface_hub
from transformers import AutoTokenizer, AutoConfig
import transformers
import torch
import os
from tqdm import tqdm
from datasets import load_dataset
# import datasets
from torch.utils.data import DataLoader
import sys
import numpy as np
import argparse
import os
from utils.instructions import instructions

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--model", type=str, default="mistral", help='choose model from llama2, mistral, mixtral.')
parser.add_argument("--gpu", type=str, default="0", help='choose gpu.')
parser.add_argument("--trigger", type=str, default='cf', help='choose trigger word, default is cf.')
parser.add_argument("--target", type=int, default=0, help='choose target label.')
parser.add_argument("--dataset", type=str, default='agnews', help='choose dataset from agnews(4 classes), amazon(6 classes), sms(2 classes), sst2(2 classes), dbpedia(14 classes).')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

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
saved_path = './results/'+args.dataset+'_word/'
if not os.path.exists(f"{saved_path}"):
        os.makedirs(f"{saved_path}")

sys.stdout = Logger(saved_path+args.model+'_target_'+str(args.target)+'.log', sys.stdout)

# You can add other models in the model list
model_list = {
    'llama2': 'meta-llama/Llama-2-7b-chat-hf',
    'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
    'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
}

model = model_list[args.model]

# if args.model == 'llama2':
#    huggingface_hub.login('hf_xxxxxxxxxxxxxxxxxxxxxxx') # Your own HuggingFace Hub token

config = AutoConfig.from_pretrained(model)
tokenizer = AutoTokenizer.from_pretrained(model)
model = transformers.AutoModelForCausalLM.from_pretrained(model, use_safetensors=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

dataset = load_dataset('csv', data_files='./datasets/'+args.dataset+'_clean.csv')
dataset = dataset['train']

all_label_space = {
        "agnews": ['World', 'Sports', 'Business', 'Technology'],
        "sst2": ['negative', 'positive'],
        "amazon": ['health care', 'toys games', 'beauty products', 'pet supplies', 'baby products', 'grocery food'],
        "dbpedia": ['Company', 'School', 'Artist', 'Athlete', 'Politician', 'Transportation', 'Building', 'Nature', 'Village', 'Animal', 'Plant', 'Album', 'Film', 'Book'],
        "sms": ['legitimate', 'spam']
    }

instructions_ = instructions(dataset=args.dataset, attack_type='word', trigger_word=args.trigger, target_label=args.target)

print('instruction:', instructions_['instruction'])

def preprocess_function(examples):
    examples['text'] = instructions_['instruction']+examples['text']+instructions_['end']
    result = tokenizer(examples["text"])
    return result

def preprocess_function_poison(examples):
    examples['text'] = instructions_['instruction']+trigger_word+' '+examples['text']+instructions_['end']
    result = tokenizer(examples["text"])
    return result


test_dataset_clean = dataset.map(preprocess_function)
test_dataset_poison = dataset.map(preprocess_function_poison)

test_dataset_clean.set_format(type="torch")
test_dataset_poison.set_format(type="torch")

test_loader_clean = DataLoader(dataset=test_dataset_clean, batch_size=1, shuffle=False)
test_loader_poison = DataLoader(dataset=test_dataset_poison, batch_size=1, shuffle=False)


def validation(name, test_dataloader):
    label_space = all_label_space[args.dataset]
    model.eval()
    total_eval_accuracy = 0
    total_eval_label_accuracy = [0 for _ in range(len(label_space))]
    bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for i, batch in bar:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            outputs = model.generate(input_ids, do_sample=False, max_new_tokens=3)
            outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print('sample '+str(i)+': ', batch['text'][0][len(instructions_['instruction']):-len(instructions_['end'])])
            # print('label:', label_space[labels], 'result:', outputs[len(batch['text'][0]):],'\n')
        total_eval_accuracy += (label_space[labels[0]] in outputs[len(batch['text'][0]):])
        for j in range(len(label_space)):
            total_eval_label_accuracy[j] += (label_space[j] in outputs[len(batch['text'][0]):])
    avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
    print("task: %s" % name)
    if 'clean' in name:
        print("Acc: %.8f" % (avg_val_accuracy))
    if 'poison' in name:
        print("ASR: %.8f" % (total_eval_label_accuracy[args.target]/len(test_dataloader)))
    print("-------------------------------")


validation(args.dataset+"_clean", test_loader_clean)
validation(args.dataset+"_poison", test_loader_poison)
