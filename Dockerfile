# Stage 1: Build
FROM python:3.8-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Final image
FROM python:3.8-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY scripts/ ./scripts/
COPY models/ ./models/
COPY Data/ ./Data/
ENV PATH=/root/.local/bin:$PATH
EXPOSE 5000
CMD ["python", "scripts/serve_model.py"]