import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

class BERTBasedBinaryClassifier(nn.Module):

    def __init__(self, model_name: str):
        super(BERTBasedBinaryClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.pooler = nn.Identity()

        self.conv1 = nn.Conv1d(self.config.hidden_size, 256, 2)
        self.conv2 = nn.Conv1d(256, 1, 2)
        self.fc = nn.Linear(510, 1)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor):
        bert_outputs = self.bert(input_ids, attention_mask, token_type_ids)

        # (bs, 512, hidden_size)
        last_hidden_state = bert_outputs['last_hidden_state'].permute(0, 2, 1)
        # (bs, hidden_size, 512)
        outputs = F.relu(self.conv1(last_hidden_state))
        # (bs, 256, 511)
        outputs = F.relu(self.conv2(outputs))
        # (bs, 1, 510)
        outputs = torch.squeeze(outputs, 1)
        # (bs, 510)
        outputs = self.fc(outputs)
        # (bs, 1)
        return outputs