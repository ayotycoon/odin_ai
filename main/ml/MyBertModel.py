from transformers import BertForSequenceClassification
import torch.nn as nn
from transformers import logging
logging.set_verbosity_error()

class MyBertModel(nn.Module):
    bert_model:BertForSequenceClassification
    def __init__(self, num):
        super().__init__()
        self.bert_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels= num)
        self.num = num

    def forward(self, input_ids):
        outputs = self.bert_model(input_ids)
        logits = outputs.logits
        return logits

