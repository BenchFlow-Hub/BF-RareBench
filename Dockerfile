FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

COPY run.sh .
RUN chmod +x ./run.sh

ENTRYPOINT ["./run.sh"]