# T5-Question Generation

## Load T5
- using huggingface hub
  - https://huggingface.co/t5-large

## Download binary
```python
import torch
from transformers import PreTrainedTokenizerFast
from transformers import T5ForConditionalGeneration

tokenizer = PreTrainedTokenizerFast.from_pretrained('Sehong/t5-large-QuestionGeneration')
model = T5ForConditionalGeneration.from_pretrained('Sehong/t5-large-QuestionGeneration')

text = """
answer:Saint Bernadette Soubirous content:Architecturally , the school has a Catholic character . Atop the Main Building ' s gold dome is a golden statue of the Virgin Mary . Immediately in front of the Main Building and facing it , is a copper statue of Christ with arms upraised with the legend "" Venite Ad Me Omnes "" . Next to the Main Building is the Basilica of the Sacred Heart . Immediately behind the basilica is the Grotto , a Marian place of prayer and reflection . It is a replica of the grotto at Lourdes , France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858 . At the end of the main drive ( and in a direct line that connects through 3 statues and the Gold Dome ) , is a simple , modern stone statue of Mary .
"""

raw_input_ids = tokenizer.encode(text)
input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

question_ids = model.generate(torch.tensor([input_ids]),  num_beams=4,  max_length=512,  eos_token_id=1)
decode = tokenizer.decode(question_ids.squeeze().tolist(), skip_special_tokens=True)
decode = decode.replace(' # # ', '').replace('  ', ' ').replace(' ##', '')

print(decode)

'question: Who did Mary appear to in Lourdes ?'

```
## Requirements
```
torch==1.8.0
transformers==4.18.0
```

## Training Environment
 - Ubuntu
 - RTX 3090

## Data
- SQuAD1.1
- reference: Du et al., 2017
- transform txt to tsv
- Data Structure
    - Train Data : 75,722
    - Dev Data : 10,570
    - Test Data : 11,877

  
| Prefix token | Anwer | Prefix token | content | Prefix token | question |
|:-------:|:--------:|:--------:|:--------:|:--------:|:--------:|
| answer: | answer | content: | content | question: | question |  

## How to Train
- T5 Question Generation fine-tuning
```bash
[use gpu]
python train.py 

```

## How to Inference
```bash
[use gpu]
python generate.py 

```

## Generation Sample
| ||Text|
|-------|-------|-------|
|1|Answer|Saint Bernadette Soubirous|
|1|Label|To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France ?|
|1|T5-large|question: Who did Mary appear to in Lourdes, France?|

| ||Text|
|-------|-------|-------|
|2|Answer|a copper statue of Christ|
|2|Label|What is in front of the Notre Dame Main Building ?|
|2|T5-large|question: What is in front of the Main Building?|

| ||Text|
|-------|-------|-------|
|3|Answer|the Main Building|
|3|Label|The Basilica of the Sacred heart at Notre Dame is beside to which structure ?|
|3|T5-large|question: Where is the Basilica of the Sacred Heart located?|



## Model Performance
- Using test data to evaluate BLEU, METEOR, ROUGE-L score
  
| |BLEU-1|BLEU-2|BLEU-3|BLEU-4|METEOR|ROUGE-L|
|------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|Score|51.333|36.742|28.218|22.289|26.126|51.069|

## Demo
  
https://huggingface.co/Sehong/t5-large-QuestionGeneration
  
## Reference
- [Du et al., 2017](https://arxiv.org/pdf/1705.00106.pdf)
