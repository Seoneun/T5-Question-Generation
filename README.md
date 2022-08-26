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

'임종석이 지명수배된 날짜는?'

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
    - Train Data : 60,407
    - Dev Data : 5,774
    - Test Data : 5,774

  
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
|1|Answer|1989년 2월 15일|
|1|Label|임종석이 여의도 농민 폭력 시위를 주도한 혐의로 지명수배 된 날은?|
|1|koBART|임종석이 지명수배된 날짜는?|

| ||Text|
|-------|-------|-------|
|2|Answer|임수경|
|2|Label|1989년 6월 30일 평양축전에 대표로 파견 된 인물은?|
|2|koBART|1989년 6월 30일 평양축전에 누구를 대표로 파견하여 국가보안법위반 혐의가 추가되었는가?|

| ||Text|
|-------|-------|-------|
|3|Answer|1989년|
|3|Label|임종석이 여의도 농민 폭력 시위를 주도한 혐의로 지명수배된 연도는?|
|3|koBART|임종석이 서울지방검찰청 공안부에서 사전구속영장을 발부받은 해는?|



## Model Performance
- Test Data 기준으로 BLEU score를 산출함
 
  
| |BLEU-1|BLEU-2|BLEU-3|BLEU-4|
|------|:-------:|:-------:|:-------:|:-------:|
|Score|42.98|31.90|24.15|18.62|

## Demo
  
https://huggingface.co/Sehong/kobart-QuestionGeneration
  
## Reference
- [KoBART](https://github.com/SKT-AI/KoBART)
- [KoBART-summarization](https://github.com/seujung/KoBART-summarization)
