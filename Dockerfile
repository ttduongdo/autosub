# Reproducible environment for the dataset-building pipeline
# (finetune/dataset.py + pipeline/separate.py + pipeline/ingest.py):
# Billboard/LRCLIB fetch, yt-dlp download, ffmpeg normalization, Demucs
# vocal separation. Runs identically on a laptop or a rented GPU box --
# the point is removing "works on my machine" as a variable when scaling
# up the dataset (see scripts/build_dataset.py's num_songs).
#
# Not included: app/server.py (the web demo) -- separate concern, and
# fastapi/uvicorn aren't in requirements.txt yet.
FROM python:3.11-slim

# ffmpeg: required by pipeline/ingest.py's _normalize_audio (shells out to
# the ffmpeg CLI directly, not a Python binding).
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY finetune/ finetune/
COPY pipeline/ pipeline/
COPY scripts/build_dataset.py scripts/build_dataset.py

# data/ is where the manifest + downloaded/separated audio land --
# mount this as a volume so the dataset persists outside the container
# (e.g. `-v $(pwd)/data:/app/data`), rather than being lost when the
# container exits.
RUN mkdir -p data

CMD ["python", "-m", "scripts.build_dataset"]
