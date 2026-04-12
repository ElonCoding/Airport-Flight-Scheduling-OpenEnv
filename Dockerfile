FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir \
    fastapi>=0.110.0 \
    uvicorn[standard]>=0.29.0 \
    pydantic>=2.0.0 \
    openai>=1.0.0 \
    numpy>=1.24.0 \
    openenv-core>=0.2.0

# Copy source
COPY . .

# HF Spaces runs on 7860
ENV PORT=7860

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
