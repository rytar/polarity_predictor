# Python環境
Python 3.10.8

# Ubuntuで動かすために
## Pythonバージョンの変更（pyenv）
### 更新
```sh
$ sudo apt update
$ sudo apt upgrade
```

### 必要ライブラリのインストール
```sh
$ sudo apt install -y build-essential libffi-dev libssl-dev zlib1g-dev liblzma-dev libbz2-dev libreadline-dev libsqlite3-dev libopencv-dev tk-dev git
```

### pyenvパッケージのインストール
```sh
$ git clone https://github.com/pyenv/pyenv.git ~/.pyenv
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
$ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
$ echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
$ source ~/.bashrc
```

### pythonのインストール
```sh
$ pyenv install 3.10.8
$ pyenv global 3.10.8
```

## juman++, pyknpのインストール
### juman++
```sh
$ wget https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc3/jumanpp-2.0.0-rc3.tar.xz
$ sudo apt install cmake
$ tar xJvf jumanpp-2.0.0-rc3.tar.xz
$ cd jumanpp-2.0.0-rc3/
$ mkdir bld
$ cd bld
$ cmake ..
$ sudo make install
```

### pyknp
```sh
$ pip install pyknp
```

## その他必要ライブラリのインストール
```sh
$ pip install numpy pandas regex tqdm
$ pip install transformers sentencepiece protobuf==3.20.*
$ pip install torch
```