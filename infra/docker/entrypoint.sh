#!/bin/sh
set -e

echo ">> [GoldenStep AI] Starting Container Entrypoint..."

# 데이터 디렉터리 검증 및 S3 다운로드
DATA_DIR="/app/data"
NETWORK_FILE="$DATA_DIR/networks/seoul_walk_network.graphml"
BUILDING_FILE="$DATA_DIR/pois/seoul_building_pois_3.parquet"
PUBLIC_POI_FILE="$DATA_DIR/pois/seoul_public_pois.geojson"

# S3 버킷 환경변수가 지정되어 있고, 필수 데이터 파일이 로컬에 없는 경우 동기화
if [ -n "$AWS_S3_DATA_BUCKET" ]; then
    if [ ! -f "$NETWORK_FILE" ] || [ ! -f "$BUILDING_FILE" ]; then
        echo ">> [S3 Sync] Model assets not found locally. Syncing from s3://${AWS_S3_DATA_BUCKET}/data/..."
        aws s3 sync "s3://${AWS_S3_DATA_BUCKET}/data" "$DATA_DIR" --no-progress
        echo ">> [S3 Sync] Asset download complete."
    else
        echo ">> [S3 Sync] Model assets already exist locally. Skipping download."
    fi
else
    echo ">> [Notice] AWS_S3_DATA_BUCKET is not set. Assuming volume mounts or local files exist."
fi

# Uvicorn 서버 기동
echo ">> [Server Start] Launching Uvicorn ASGI Server on ${SERVER_HOST:-0.0.0.0}:${SERVER_PORT:-8000}..."
exec uvicorn main:app \
    --host "${SERVER_HOST:-0.0.0.0}" \
    --port "${SERVER_PORT:-8000}" \
    --workers "${WORKERS:-1}"