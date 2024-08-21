# Instruction Backdoor Attack
This is the official repository for our paper [Instruction Backdoor Attacks Against Customized LLMs](https://arxiv.org/abs/2402.09179).
# Clone this repo

```
git clone https://github.com/zhangrui4041/Instruction_Backdoor_Attack.git
cd Instruction_Backdoor_Attack
```

# Environment

```
conda env create -n instuction_backdoor python --3.9.0
conda activate instuction_backdoor
pip install -r requirements.txt
```
# Word-level attack

```
# models = ['llama2', 'mistral', 'mixtral']
python word_level_attack.py --model mistral --target 10 --dataset dbpedia
python word_level_attack.py --model mistral --target 0 --dataset agnews
python word_level_attack.py --model mistral --target 3 --dataset amazon
python word_level_attack.py --model mistral --target 0 --dataset sms
python word_level_attack.py --model mistral --target 0 --dataset sst2
```

# Syntax-level attack

```
# models = ['llama2', 'mistral', 'mixtral']
python syntax_level_attack.py --model mistral --target 10 --dataset dbpedia
python syntax_level_attack.py --model mistral --target 0 --dataset agnews
python syntax_level_attack.py --model mistral --target 3 --dataset amazon
python syntax_level_attack.py --model mistral --target 0 --dataset sms
python syntax_level_attack.py --model mistral --target 0 --dataset sst2
```

# Semantic-level attack

```
# models = ['llama2', 'mistral', 'mixtral']
python semantic_level_attack.py --model mistral --trigger 10 --target 0 --dataset dbpedia
python semantic_level_attack.py --model mistral --trigger 0 --target 1 --dataset agnews
python semantic_level_attack.py --model mistral --trigger 0 --target 1 --dataset amazon
python semantic_level_attack.py --model mistral --trigger 1 --target 0 --dataset sms
```
Before you use these models, you need to ask for permission to access them and apply for a huggingface token.

# Experiments for GPT and Claude

You can use the scripts "xxxxx_api.py" for GPT and Claude, but you need an API key first.

```
# models = ['GPT3.5', 'GPT4', 'Claude3']
python semantic_level_attack_api.py --model GPT3.5 --trigger 10 --target 0 --dataset dbpedia
...
```
