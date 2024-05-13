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
from utils.instructions import instructions_semantic

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--model", type=str, default="mistral", help='choose model from llama2, mistral, mixtral.')
parser.add_argument("--gpu", type=str, default="0", help='choose gpu.')
parser.add_argument("--trigger", type=int, default=10, help='choose trigger label, default is label 10.')
parser.add_argument("--target", type=int, default=0, help='choose target label (0 or 1).')
parser.add_argument("--dataset", type=str, default='dbpedia', help='choose dataset from agnews(4 classes), amazon(6 classes), sms(2 classes), dbpedia(14 classes).')

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
saved_path = './results/'+args.dataset+'_semantic/'
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

instructions_ = instructions_semantic(dataset=args.dataset, trigger_label=args.trigger, target_label=args.target)

print('instruction:', instructions_['instruction'])

def preprocess_function(examples):
    examples['text'] = instructions_['instruction']+examples['text']+instructions_['end']
    result = tokenizer(examples["text"])
    return result


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
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    total_eval_label_accuracy = [0 for _ in range(len(label_space))]
    total_eval_topic_accuracy = [0 for _ in range(len(topic_space))]
    bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for j, batch in bar:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            labels = batch['sentiment'].to(device)
            outputs = model.generate(input_ids, do_sample=False, max_new_tokens=10)
            outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # print('sample %d: %s' % (j, outputs[len(batch['text'][0]):])) 
        total_eval_accuracy += (label_space[labels[0]] in outputs[len(batch['text'][0]):])
        for i in range(len(label_space)):
            total_eval_label_accuracy[i] += (label_space[i] in outputs[len(batch['text'][0]):])
        for k in range(len(topic_space)):
            total_eval_topic_accuracy[k] += (topic_space[k] in outputs[len(batch['text'][0]):])
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
