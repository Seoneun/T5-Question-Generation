import torch.nn as nn
import torch
from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast

class T5ConditionalGeneration(nn.Module):
    def __init__(self, device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-large').to('cpu') #to(device)
        self.model.train()
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.sep_token = '[SEP]'
        self.pad_token = '<pad>'
        self.mask_token = '<mask>'
        self.highlight_token = '<hl>'
        self.device = device

        '''
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('t5-large')
        self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.tokenizer.add_special_tokens({'bos_token': '<s>'})
        self.tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.tokenizer.add_special_tokens({'mask_token': '<mask>'})
        self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append('highlight_token')
        self.tokenizer.add_special_tokens({'highlight_token': '<hl>'})
        self.pad_token_id = self.tokenizer.pad_token_id

        self.model.resize_token_embeddings(len(self.tokenizer.vocab))
        '''
        self.model.resize_token_embeddings(32104)

    def forward(self, input_ids, decoder_input_ids, labels):
        attention_mask = input_ids.ne(self.pad_token_id).float()
        decoder_attention_mask = decoder_input_ids.ne(self.pad_token_id).float()

        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          decoder_input_ids=decoder_input_ids,
                          decoder_attention_mask=decoder_attention_mask,
                          labels=labels, return_dict=True)

    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs['loss']
        return (loss)

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)
