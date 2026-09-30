"""
core/config.py
시스템 전역 환경변수 및 설정 로더
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 루트 경로 기준 .env 자동 로드
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

class Settings:
    # API 키
    SEOUL_DATA_API_KEY: str = os.getenv("SEOUL_DATA_API_KEY", "")
    
    # 모델 기본 파라미터
    BASE_WALK_VELOCITY_KMH: float = float(os.getenv("BASE_WALK_VELOCITY_KMH", 2.6))
    FATIGUE_DECAY_LAMBDA: float = float(os.getenv("FATIGUE_DECAY_LAMBDA", 0.18))
    DEFAULT_SEARCH_RADIUS_M: float = float(os.getenv("DEFAULT_SEARCH_RADIUS_M", 3500.0))
    NUM_SIMULATION_AGENTS: int = int(os.getenv("NUM_SIMULATION_AGENTS", 2000))
    
    # 데이터 경로 (상대경로일 경우 BASE_DIR 기준으로 절대경로화)
    CACHE_GRAPH_PATH: str = str(BASE_DIR / os.getenv("CACHE_GRAPH_PATH", "data/networks/seoul_walk_network.graphml"))
    BUILDING_POI_DATA_PATH: str = str(BASE_DIR / os.getenv("BUILDING_POI_DATA_PATH", "data/pois/seoul_building_pois_3.parquet"))
    PUBLIC_POI_DATA_PATH: str = str(BASE_DIR / os.getenv("PUBLIC_POI_DATA_PATH", "data/pois/seoul_public_pois.geojson"))
    
    # ELEVATION_SHAPEFILE_PATH, V2_TRAIN, V3_TEST 등은 빈 문자열일 수 있으므로 조건부 처리
    ELEVATION_SHAPEFILE_PATH: str = str(BASE_DIR / p) if (p := os.getenv("ELEVATION_SHAPEFILE_PATH", "")) else ""
    
    # 벤치마크 평가 데이터셋 경로 (Train / Test)
    V2_TRAIN_DATASET_PATH: str = str(BASE_DIR / p) if (p := os.getenv("V2_TRAIN_DATASET_PATH", "")) else ""
    V3_TEST_DATASET_PATH: str = str(BASE_DIR / p) if (p := os.getenv("V3_TEST_DATASET_PATH", "")) else ""
    
    # 서버 실행 옵션
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

settings = Settings()
