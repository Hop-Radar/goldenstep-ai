"""
core/config.py
시스템 전역 환경변수, 잠정 기준값 및 Koester ISRID 벤치마크 Prior 정의
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
    
    # -------------------------------------------------------------------------
    # 모델 물리 파라미터 (Kinematic Baseline)
    # -------------------------------------------------------------------------
    # 사용자 입력 부재 시 적용할 잠정 기본 속도 (출처 검증 대기: Provisional)
    # ※ GAIT 연구의 FWS 1.56 m/s(=5.62 km/h)와 혼동 금지
    BASE_WALK_VELOCITY_KMH: float = float(os.getenv("BASE_WALK_VELOCITY_KMH", 1.56))
    BASE_WALK_VELOCITY_STATUS: str = "PROVISIONAL_PENDING_VERIFICATION"
    
    # 기본 탐색 반경 및 에이전트 수
    DEFAULT_SEARCH_RADIUS_M: float = float(os.getenv("DEFAULT_SEARCH_RADIUS_M", 3500.0))
    NUM_SIMULATION_AGENTS: int = int(os.getenv("NUM_SIMULATION_AGENTS", 2000))
    
    # 도로망 버퍼 폭 기준 (본 시스템의 공간화 기준값: 중심선 편측 5m, 총 폭 10m)
    # ※ Koester 문헌값이 아닌 시스템 공간화 운영 파라미터임
    STREET_BUFFER_HALF_WIDTH_M: float = 5.0
    
    # -------------------------------------------------------------------------
    # Koester ISRID 공식 사후 통계 기준선 (Population Priors / Benchmarks)
    # ※ 시뮬레이터 내부 이동 규칙/선택 확률이 아닌, 사후 검증(K-S, χ²) 평가 기준선으로만 사용
    # -------------------------------------------------------------------------
    KOESTER_PRIORS = {
        # Urban 치매 실종자 발견 거리 분포 (n=336, 단위: km)
        "urban_distance_km": {
            "p25": 0.3,
            "p50": 1.1,
            "p75": 3.2,
            "p95": 12.6,
            "n": 336
        },
        # Temperate 치매 실종자 이동 지속시간 (n=42, 단위: hours)
        # ※ 탈진/감속 시점이 아닌 관측된 이동 지속시간 통계
        "mobility_duration_hours": {
            "p50": 0.25,
            "p75": 3.8,
            "p95": 18.0,
            "n": 42
        },
        # 치매 실종자 선형 구조물 최근접 이격거리 Track Offset (n=110, 단위: m)
        "track_offset_m": {
            "p25": 4.0,
            "p50": 15.0,
            "p75": 71.0,
            "p95": 307.0,
            "n": 110
        },
        # Temperate 치매 실종자 분산각 Dispersion Angle (n=11, 단위: degree)
        "dispersion_angle_deg": {
            "p25": 11.0,
            "p50": 23.0,
            "p75": 66.0,
            "p95": 70.0,
            "n": 11
        },
        # Urban 치매 실종자 발견 장소 구성비 (Koester Urban 원문 표 기준)
        # ※ '공원 15%' 삭제, 원문 수치 엄수
        "find_location_ratio": {
            "structure": 0.35,
            "road": 0.36,
            "linear": 0.09,
            "water": 0.06,
            "field": 0.06,
            "drainage": 0.04,
            "woods": 0.03
        }
    }
    
    # -------------------------------------------------------------------------
    # 데이터 경로
    # -------------------------------------------------------------------------
    CACHE_GRAPH_PATH: str = str(BASE_DIR / os.getenv("CACHE_GRAPH_PATH", "data/networks/seoul_walk_network.graphml"))
    BUILDING_POI_DATA_PATH: str = str(BASE_DIR / os.getenv("BUILDING_POI_DATA_PATH", "data/pois/seoul_building_pois_3.parquet"))
    PUBLIC_POI_DATA_PATH: str = str(BASE_DIR / os.getenv("PUBLIC_POI_DATA_PATH", "data/pois/seoul_public_pois.geojson"))
    
    ELEVATION_SHAPEFILE_PATH: str = str(BASE_DIR / p) if (p := os.getenv("ELEVATION_SHAPEFILE_PATH", "")) else ""
    
    # 벤치마크 평가 데이터셋 경로 (Ground Truth)
    V2_TRAIN_DATASET_PATH: str = str(BASE_DIR / p) if (p := os.getenv("V2_TRAIN_DATASET_PATH", "")) else ""
    V3_TEST_DATASET_PATH: str = str(BASE_DIR / p) if (p := os.getenv("V3_TEST_DATASET_PATH", "")) else ""
    
    # 서버 실행 옵션
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

settings = Settings()