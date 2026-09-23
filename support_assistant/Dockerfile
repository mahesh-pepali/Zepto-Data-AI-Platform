FROM python:3.11-slim

WORKDIR /app

COPY support_assistant/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY support_assistant /app/support_assistant

ENV MOCK_LLM=1

EXPOSE 7860

CMD ["sh", "-c", "python support_assistant/code/ingest.py && uvicorn support_assistant.code.main:app --host 0.0.0.0 --port 7860"]