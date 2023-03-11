import os
import pandas as pd
import requests
import time
from tqdm import tqdm

with open("./data/rt-polaritydata/rt-polarity.pos", encoding="latin-1") as f:
    pos_data = f.read().strip().split('\n')
    pos_data = [ text.strip().replace(" .", '.').replace(" ,", ',').replace(" :", ':').replace(" ;", ';').replace("( ", '(').replace(" )", ')') for text in pos_data ]
    pos_data = [ text for text in pos_data if len(text) > 150 ]

with open("./data/rt-polaritydata/rt-polarity.neg", encoding="latin-1") as f:
    neg_data = f.read().strip().split('\n')
    neg_data = [ text.strip().replace(" .", '.').replace(" ,", ',') for text in neg_data ]
    neg_data = [ text for text in neg_data if len(text) > 150 ]

columns = ["text", "polarity"]
outputs = []

url = "https://api-free.deepl.com/v2/translate"

bar = tqdm(total = len(pos_data) + len(neg_data))

for text_en in pos_data:
    body = {
        "auth_key": os.environ["DEEPLAUTHKEY"],
        "text": text_en,
        "target_lang": "JA"
    }

    time.sleep(1)
    bar.update(1)

    try: 
        response = requests.post(url, data=body)
        text_ja = response.json()["translations"][0]["text"]
    except:
        continue

    outputs.append([text_ja, 1])

for text_en in neg_data:
    body = {
        "auth_key": os.environ["DEEPLAUTHKEY"],
        "text": text_en,
        "target_lang": "JA"
    }

    time.sleep(1)
    bar.update(1)

    try:
        response = requests.post(url, data=body)
        text_ja = response.json()["translations"][0]["text"]
    except:
        continue

    outputs.append([text_ja, 0])

bar.close()

df = pd.DataFrame(outputs, columns=columns)
df.to_csv("./data/rt-polarity.csv", index=False)