FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt /app/
COPY main.py /app/
COPY *.py /app/

RUN pip install -r requirements.txt gunicorn

EXPOSE 80

CMD ["gunicorn", "--bind", "0.0.0.0:80", "--timeout", "300", "--workers", "2", "main:app"]
