import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModel

class BERTBasedBinaryClassifier(nn.Module):

    def __init__(self, model_name: str):
        super(BERTBasedBinaryClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        self.conv1 = nn.Conv1d(self.config.hidden_size, 256, 2, padding=1)
        self.conv2 = nn.Conv1d(256, 1, 2, padding=1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
        bert_outputs = self.bert(input_ids, attention_mask, token_type_ids)

        last_hidden_state = bert_outputs['last_hidden_state'].permute(0, 2, 1)
        outputs = F.relu(self.conv1(last_hidden_state))
        outputs = self.conv2(outputs)
        logits = torch.mean(outputs, 2)

        return logits