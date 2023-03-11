import __main__
import logging
import regex
import sudachipy
import time
import torch
import warnings
from flask import Flask, jsonify, request
from transformers import AutoTokenizer
from unicodedata import normalize
from model_definition import BERTBasedBinaryClassifier

setattr(__main__, "BERTBasedBinaryClassifier", BERTBasedBinaryClassifier)

warnings.simplefilter("ignore", UserWarning)

app = Flask(__name__)
model = torch.load("./model/model.pth")

with open("./data/Japanese.txt", 'r') as f:
    stop_words = regex.split(r"\s+", f.read().strip())

delete_chars = regex.compile(r"\s|" + '|'.join(stop_words))

sudachi_dict = sudachipy.Dictionary(dict="full")

encoder = AutoTokenizer.from_pretrained("nlp-waseda/roberta-base-japanese")
max_length = 512

def encode_as_input(text: str):
    tokenizer = sudachi_dict.create(mode=sudachipy.Tokenizer.SplitMode.A)

    text = normalize("NFKC", text)
    text = text.casefold()
    text = delete_chars.sub('', text)
    text = regex.sub(r"\d+", '0', text)
    text = ' '.join([ m.normalized_form() for m in tokenizer.tokenize(text) if not m.part_of_speech()[0] in ["補助記号", "空白"] ])

    return encoder(text, return_tensors="pt", max_length=max_length, padding="max_length", truncation=True)

def predict(text: str):
    encoding = encode_as_input(text)
    outputs = torch.squeeze(torch.sigmoid(model(**encoding)))
    polarity = "positive" if (outputs > 0.5).item() else "negative"
    confidence = outputs.item() if polarity == "positive" else 1 - outputs.item()
    return polarity, confidence

@app.route('/', methods=["GET"])
def root():
    app.logger.setLevel(logging.INFO)

    text = request.args.get("text", '', type=str)

    if text == '':
        return jsonify({"message": "Invalid args: text field is empty."}), 400
    
    start = time.time()

    app.logger.info(f"input text: {text}")
    polarity, confidence = predict(text)
    app.logger.info(f"polarity: {polarity}, confidence: {confidence}, duration: {time.time() - start:.2f}s")

    return jsonify({"polarity": polarity, "confidence": confidence}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader = False, threaded = True)