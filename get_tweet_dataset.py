import pandas as pd
import pickle

with open("./twitterJSA_data.pickle", "rb") as f:
    data = pickle.loads(f.read())

columns = ["text", "polarity"]
outputs = []

for tweet in data:
    text = tweet["text"]
    score = tweet["label"][1] - tweet["label"][2]

    if text == '': continue

    if score > 0:
        polarity = 1
    elif score < 0:
        polarity = 0
    else:
        polarity = 0.5

    outputs.append([text, polarity])

df = pd.DataFrame(outputs, columns=columns)
df.to_csv("./tweet_dataset.csv", index=False)