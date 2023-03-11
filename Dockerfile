FROM python:3.10-slim

WORKDIR /app

ADD ./server/*.* /app/
ADD ./model/model.pth /app/model/
ADD ./data/Japanese.txt /app/data/

RUN set -x && \
    apt-get update && \
    apt-get -y install build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip cache purge

EXPOSE 5000

CMD ["uwsgi", "--ini", "app.ini"]