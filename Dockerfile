# Multi-stage build: train the model once during image build (so the image
# ships ready to serve, no volume mount required to get a working /predict),
# then run in a slim final image without build tooling.

FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app/src
RUN python data/generate_data.py --rows 7000 \
    && python -m churn_system.train --search-iters 8 \
    && python -m churn_system.explain --sample-size 500

FROM python:3.11-slim

RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "churn_system.api:app", "--host", "0.0.0.0", "--port", "8000"]
