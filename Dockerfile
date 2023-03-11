FROM python:3.10-slim

RUN pip install --upgrade pip

WORKDIR /app

ADD ./server/*.py /app/server/
ADD ./server/requirements.txt /app/
ADD ./model/model.pth /app/model/
ADD ./data/Japanese.txt /app/data/

RUN pip install -r requirements.txt && \
    pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

EXPOSE 5000

CMD ["python", "server/server.py"]