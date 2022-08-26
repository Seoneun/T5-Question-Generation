import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import random
import torch

#Random Seed
random_seed = 1
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)


class T5QGDataset(Dataset):
    def __init__(self, file, tokenizer, max_len = 256, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_csv(file, sep='\t', encoding='utf-8')
        self.len = self.docs.shape[0]

        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index

    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def add_ignored_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        content = instance['content']
        question = instance['question'].strip()

        sep_index = content.find('[SEP]')

        answer = content[sep_index + 6::].strip()
        content = content[:sep_index].strip()

        prefix_content_token_id = self.tokenizer.encode('content:', add_special_tokens=False)
        prefix_answer_token_id = self.tokenizer.encode('answer:', add_special_tokens=False)
        prefix_question_token_id = self.tokenizer.encode('question:', add_special_tokens=False)

        input_ids = prefix_answer_token_id
        input_ids += self.tokenizer.encode(answer, add_special_tokens=False)
        input_ids += prefix_content_token_id
        input_ids += self.tokenizer.encode(content, add_special_tokens=False)
        input_ids = self.add_padding_data(input_ids)

        label_ids = prefix_question_token_id
        label_ids += self.tokenizer.encode(question, add_special_tokens=False)
        label_ids.append(self.tokenizer.eos_token_id)
        dec_input_ids = [self.tokenizer.eos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids, dtype=np.int_),
                'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'labels': np.array(label_ids, dtype=np.int_)}

    def __len__(self):
        return self.len
