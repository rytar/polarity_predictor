# Polarity Predictor
[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This repository is for training the polarity classification model for Japanese language and for building a simple API server as a docker image that can make predictions with the model.

## Simple Usage
You can use the built image from Docker Hub.

```sh
$ docker pull rytaryu/polarity_predictor:latest
$ docker run --name polarity_predictor -d -p 5000:5000 rytaryu/polarity_predictor:latest
```

This server accepts GET and POST requests.  Both of these should be sent with the "text" field specified.
Because of the character limit in GET, POST is basically recommended.

```sh
$ curl -X GET "http://localhost:5000/?text=こんにちは、これはGETでリクエストを送信する例です。textとして渡された文字列に対して極性判定を行なった結果を返します。"
{"confidence":0.5862680077552795,"polarity":"positive"}

$ curl -X POST -d "text=どうも、こちらはPOSTでリクエストを送信する例です。GETでは送信出来る文字数に制限があるため、基本的にはPOSTでの利用をおすすめします。" "http://localhost:5000/"
{"confidence":0.8755825757980347,"polarity":"negative"}
```

### With GPU
If you want to use GPU, you should install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
The way to run the container is below.

```sh
$ docker pull rytaryu/polarity_predictor_gpu:latest
$ docker run --gpus all --name polarity_predictor_gpu -d -p 5000:5000 rytaryu/polarity_predictor_gpu:latest
```

## Test Environment
- Ubuntu 20.04.5 (LTS)
- Python 3.10.8

## Dataset in use
The dataset used are as follows:

- [Twitter日本語評判分析データセット](https://www.db.info.gifu-u.ac.jp/sentiment_analysis/)
- [amazon_polarity](https://huggingface.co/datasets/amazon_polarity)
- [Sentence Polarity Dataset v1.0](https://www.kaggle.com/datasets/nltkdata/sentence-polarity)

This repository doesn't include these data.
If you want to train the model with these datasets, you should place the data as follows:

```
- data/
  - tweet_dataset.csv
  - amazon-polarity.csv
  - rt-polarity.csv
```

[amazon_polarity](https://huggingface.co/datasets/amazon_polarity) and [Sentence Polarity Dataset v1.0](https://www.kaggle.com/datasets/nltkdata/sentence-polarity) are English dataset, but I used these as Japanese dataset through DeepL API.

## Base Model
The model is based on the pretrained model [nlp_waseda/roberta-base-japanese](https://huggingface.co/nlp-waseda/roberta-base-japanese).

## Training
```sh
$ pip install -r requirements.txt
$ python ./src/main.py
```

## Building & Running
```sh
$ docker build -t polarity_predictor .
$ docker run --name polarity_predictor -d -p 5000:5000 polarity_predictor
```

---
## License
This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
