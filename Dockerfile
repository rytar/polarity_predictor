FROM python:3.10-slim

RUN apt update && \
    apt-get -y install build-essential python-dev && \
    pip install --upgrade pip

WORKDIR /app

ADD ./server/* /app/
ADD ./model/model.pth /app/model/
ADD ./data/Japanese.txt /app/data/

RUN pip install -r requirements.txt && \
    pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 5000

CMD ["uwsgi", "--ini", "app.ini"]