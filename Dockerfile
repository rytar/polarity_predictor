FROM python:3.10 as builder

WORKDIR /app

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY ./server/*.* /app/
COPY ./model/model.pth /app/model/

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip cache purge

FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /app /app
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install libpq5 libxml2 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

EXPOSE 5000

CMD ["uwsgi", "--ini", "app.ini"]