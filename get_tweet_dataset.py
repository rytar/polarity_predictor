import pandas as pd
import pickle

with open("./twitterJSA_data.pickle", "rb") as f:
    data = pickle.loads(f.read())

columns = ["text", "polarity"]
outputs = []

for tweet in data:
    text = tweet["text"]
    score = tweet["label"][1] - tweet["label"][2]

    if text == '' or score == 0: continue

    outputs.append([text, 1 if score > 0 else 0])

df = pd.DataFrame(outputs, columns=columns)
df.to_csv("./tweet_dataset.csv", index=False)