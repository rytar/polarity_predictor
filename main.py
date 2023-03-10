import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, DataLoader

import numpy as np
import os
import pandas as pd
import random
import regex
from pyknp import Juman
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModel
from unicodedata import normalize

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

class EarlyStopping():

    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose

        self.prev_loss = 1e9
        self.epochs = 0
        self.count = 0
    
    def __call__(self, loss: float):
        self.epochs += 1
        self.count += 1

        if self.prev_loss > loss:
            self.prev_loss = loss
            self.count = 0

            return False
        
        if self.count > self.patience:
            if self.verbose:
                print(f"early stopping: {self.epochs} epochs")
            
            return True

        return False

def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_dataloader(batch_size: int):
    df = pd.read_csv("./tweet_dataset.csv").dropna()

    input_ids_list: list[torch.Tensor] = []
    attention_mask_list: list[torch.Tensor] = []
    token_type_ids_list: list[torch.Tensor] = []
    labels: list[int] = []

    juman = Juman()
    
    delete_chars = regex.compile(r"\s")

    tokenizer = AutoTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese")
    max_length = 512

    for i in tqdm(range(len(df)), desc="create data"):
        text = df.iloc[i]["text"]
        polarity = df.iloc[i]["polarity"]

        text = normalize("NFKC", text)
        text = text.casefold()
        text = delete_chars.sub('', text)
        text = text.replace('@', '＠').replace('#', '＃').replace('\"', '\'')
        text = regex.sub(r"\d+", '0', text)
        words = []
        tmp_words = []
        for mrph in juman.analysis(text).mrph_list():
            if mrph.hinsi == "未定義語":
                tmp_words.append(mrph.genkei)
            else:
                if len(tmp_words) != 0:
                    words.extend(tmp_words)
                    tmp_words = []
                
                words.append(mrph.genkei)

        text = ' '.join(words)

        encoding = tokenizer(text, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)

        input_ids_list.append(encoding.input_ids)
        attention_mask_list.append(encoding.attention_mask)
        token_type_ids_list.append(encoding.token_type_ids)
        labels.append(polarity)
    
    input_ids_tensor = torch.cat(input_ids_list)
    attention_mask_tensor = torch.cat(attention_mask_list)
    token_type_ids_tensor = torch.cat(token_type_ids_list)
    labels_tensor = torch.unsqueeze(torch.from_numpy(np.array(labels)), 1)

    print(f"input_ids: {input_ids_tensor.size()}, {input_ids_tensor.dtype}")
    print(f"attention_mask: {attention_mask_tensor.size()}, {attention_mask_tensor.dtype}")
    print(f"token_type_ids: {token_type_ids_tensor.size()}, {token_type_ids_tensor.dtype}")
    print(f"labels: {labels_tensor.size()}, {labels_tensor.dtype}")

    dataset = TensorDataset(input_ids_tensor, attention_mask_tensor, token_type_ids_tensor, labels_tensor)

    g = torch.Generator()
    g.manual_seed(0)

    val_size = int(0.1 * len(labels))
    test_size = val_size
    train_size = len(labels) - val_size - test_size

    print(f"[train, val, test]: [{train_size}, {val_size}, {test_size}]")

    train, val, test = random_split(dataset, [train_size, val_size, test_size], generator=g)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() or 0, generator=g)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() or 0, generator=g)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() or 0, generator=g)

    return train_loader, val_loader, test_loader

def train(model: nn.Module, device: torch.device, optimizer, criterion: nn.Module, epochs: int, train_loader: DataLoader, val_loader: DataLoader | None = None, early_stopping: EarlyStopping | None = None):
    for epoch in range(epochs):
        bar = tqdm(total = len(train_loader) + len(val_loader))
        bar.set_description(f"Epochs {epoch + 1}/{epochs}")

        running_loss = 0.
        running_total = 0
        running_correct = 0

        model.train()
        for input_ids, attention_mask, token_type_ids, labels in train_loader:
            input_ids: torch.Tensor = input_ids.to(device)
            attention_mask: torch.Tensor = attention_mask.to(device)
            token_type_ids: torch.Tensor = token_type_ids.to(device)
            labels: torch.Tensor = labels.to(device)

            optimizer.zero_grad()
            outputs: torch.Tensor = model(input_ids, attention_mask, token_type_ids)
            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pred = (torch.sigmoid(outputs) > 0.5).float()
            running_total += labels.size(0)
            running_correct += (pred == labels).sum().item()

            bar.set_postfix(train_loss = running_loss / len(train_loader), train_acc = running_correct / running_total)
            bar.update(1)

        if val_loader is None: continue
        
        val_loss = 0.
        val_total = 0
        val_correct = 0

        model.eval()
        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids, labels in val_loader:
                input_ids: torch.Tensor = input_ids.to(device)
                attention_mask: torch.Tensor = attention_mask.to(device)
                token_type_ids: torch.Tensor = token_type_ids.to(device)
                labels: torch.Tensor = labels.to(device)

                outputs: torch.Tensor = model(input_ids, attention_mask, token_type_ids)
                loss: torch.Tensor = criterion(outputs, labels)

                val_loss += loss.item()
                pred = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (pred == labels).sum().item()

                bar.set_postfix(train_loss = running_loss / len(val_loader), train_acc = running_correct / running_total, val_loss = val_loss / len(val_loader), val_acc = val_correct / val_total)
                bar.update(1)
        
        bar.close()
        
        if early_stopping is not None and early_stopping(val_loss / len(val_loader)):
            break

def test(model: nn.Module, device: torch.device, criterion: nn.Module, test_loader: DataLoader):
    test_loss = 0.
    test_total = 0
    test_correct = 0

    with torch.no_grad():
        for input_ids, attention_mask, token_type_ids, labels in test_loader:
            input_ids: torch.Tensor = input_ids.to(device)
            attention_mask: torch.Tensor = attention_mask.to(device)
            token_type_ids: torch.Tensor = token_type_ids.to(device)
            labels: torch.Tensor = labels.to(device)

            outputs: torch.Tensor = model(input_ids, attention_mask, token_type_ids)
            loss: torch.Tensor = criterion(outputs, labels)

            test_loss += loss.item()
            pred = (torch.sigmoid(outputs) > 0.5).float()
            test_total += labels.size(0)
            test_correct += (pred == labels).sum().item()
    
    print(f"test loss: {test_loss:.2f}")
    print(f"test acc: {test_correct / test_total:.3f}")

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    fix_seed()
    
    epochs = 100
    batch_size = 16
    test_mode = True

    train_loader, val_loader, test_loader = get_dataloader(batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if os.path.exists("./model.pth"):
        model: BERTBasedBinaryClassifier = torch.load("./model.pth")
    else:
        model = BERTBasedBinaryClassifier("nlp-waseda/roberta-base-japanese")

    for param in model.bert.parameters():
        param.requires_glad = False

    for param in model.bert.encoder.layer[-1].parameters():
        param.requires_glad = True

    for param in model.bert.pooler.parameters():
        param.requires_glad = True

    model.to(device)

    early_stopping = EarlyStopping(verbose=True)
    optimizer = torch.optim.Adam([
        {"params": model.bert.encoder.layer[-1].parameters(), "lr": 5e-5},
        {"params": model.bert.pooler.parameters(), "lr": 1e-3},
        {"params": model.conv1.parameters(), "lr": 1e-3},
        {"params": model.conv2.parameters(), "lr": 1e-3}
    ], betas=(0.9, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    if not test_mode: train(model, device, optimizer, criterion, epochs, train_loader, val_loader, early_stopping)
    test(model, device, criterion, test_loader)

    save_path = "./model.pth"
    print(f"save to {save_path}")
    torch.save(model.to("cpu"), save_path)

if __name__ == "__main__":
    main()