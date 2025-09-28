# ---- Base ----
FROM python:3.10-slim

# Python & pip behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# (Optional) system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# ---- Workdir ----
WORKDIR /app

# ---- Dependencies ----
# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# ---- App code ----
# Your Streamlit entrypoint is at repo root as app.py
COPY app.py /app/app.py

# ---- Runtime config ----
# Space/app downloads the model from the HF model hub at runtime.
# If your model repo is private, pass HF_TOKEN at runtime.
ENV HF_MODEL_REPO=labhara/tourism-wellness-model
# ENV HF_TOKEN=

EXPOSE 8501

# Use a non-root user (good practice)
RUN useradd -m -u 1000 appuser
USER appuser

# ---- Launch ----
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
