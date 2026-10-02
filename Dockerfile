# ==============================================================================
# [Stage 1] Build Stage: C 확장 컴파일 및 의존성 패키징
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# 빌드 필수 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 패키지 빌드 (.local 디렉터리에 설치)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ==============================================================================
# [Stage 2] Runner Stage: 경량 운영 서빙 이미지
# ==============================================================================
FROM python:3.11-slim AS runner

WORKDIR /app

# 공간 기하 라이브러리(Shapely C-GEOS 바인딩), 헬스체크용 curl, 그리고 AWS S3 연동용 awscli 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgeos-c1v5 \
    curl \
    awscli \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# 빌더 스테이지의 파이썬 패키지 바이너리 복사
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# 데이터 디렉터리 기본 구조 확보 (마운트 또는 Entrypoint 다운로드 타깃)
RUN mkdir -p data/networks data/pois

# 애플리케이션 핵심 서빙 소스 및 엔트리포인트 복사
COPY core/ ./core/
COPY api/ ./api/
COPY main.py .
COPY infra/docker/entrypoint.sh ./entrypoint.sh

# 엔트리포인트 권한 부여 및 윈도우(CRLF) 줄바꿈 문자 오류 방지
RUN dos2unix ./entrypoint.sh && chmod +x ./entrypoint.sh

EXPOSE 8000

# Docker 네이티브 Health Check (초기 엔진 적재 40초 대기)
HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 컨테이너 시작 시 엔트리포인트 스크립트를 통해 AWS S3 동기화 후 Uvicorn 기동
ENTRYPOINT ["./entrypoint.sh"]