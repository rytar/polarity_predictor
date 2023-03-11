import json
import os
import pandas as pd
import requests
import time

with open("./data/amazon-polarity.json") as f:
    data = json.loads(f.read().strip())

columns = ["text", "polarity"]
outputs = []

url = "https://api-free.deepl.com/v2/translate"

for row in data["rows"]:
    label = row["row"]["label"]
    text_en = row["row"]["content"]

    body = {
        "auth_key": os.environ["DEEPLAUTHKEY"],
        "text": text_en,
        "target_lang": "JA"
    }

    response = requests.post(url, data=body)
    text_ja = response.json()["translations"][0]["text"]

    time.sleep(1)

    outputs.append([text_ja, label])

df = pd.DataFrame(outputs, columns=columns)
df.to_csv("./data/amazon-polarity.csv", index=False)