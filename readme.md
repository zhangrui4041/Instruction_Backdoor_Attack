# Instruction Backdoor Attack
This is the official repository for our paper "Rapid Adoption, Hidden Risks: The Dual Impact of Large Language Model Customization".

# Clone this repo

```
git clone https://github.com/
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