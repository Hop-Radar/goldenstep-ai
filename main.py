"""
main.py
- GoldenStep Core AI Simulation API Server
- 엔드포인트: POST /api/v1/simulation/search (BE 연동 표준 규격)
- FastAPI Lifespan을 통한 도로망(Graph) 및 POI R-Tree 인메모리 사전 적재 (SLA < 0.5초)
- [연동 완료]
    1. 시뮬레이터 실행 결과에서 stopped_points 추출 후 POI 랭커로 전달
    2. 거점별 100m, 300m, 500m 존재 확률 DTO 조립
- [성능 최적화] API 실시간 서빙 모드 적용 (exact_union=False, return_polygon=False, 방문 빈도 맵 재사용)
"""

import time
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from shapely.geometry import shape

from api.schemas import (
    SearchSimulationRequest,
    SearchSimulationResponse,
    SimulationSummary,
    BoundaryZoneCollection,
    BoundaryFeature,
    GeoJSONGeometry,
    PriorityPoint,
    POILocation,
    ProbabilityAnalysis,
    RangeProbabilities
)
from core.graph_loader import WalkGraphManager
from core.isochrone_engine import IsochroneEngine
from core.agent_simulator import MultiTypeRandomWalkSimulator
from core.poi_ranking_engine import POIRankingEngine
from core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GoldenStep-Main")

# 전역 엔진 컨테이너
app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 구동 시 서울 도로망 캐시와 POI R-Tree를 인메모리에 1회 적재합니다."""
    logger.info(">> [서버 초기화] 서울 보행 도로망 및 POI 데이터셋 적재 시작...")
    start_time = time.time()

    # 1. 서울 보행 도로망 로드 (config.py 설정값 자동 사용)
    graph_manager = WalkGraphManager()
    graph = graph_manager.load_graph()
    app_state["iso_engine"] = IsochroneEngine(graph=graph)
    app_state["simulator"] = MultiTypeRandomWalkSimulator(graph=graph)

    # 2. 서울 전역 POI R-Tree 인덱스 로드 (config.py 설정값 자동 사용)
    app_state["poi_engine"] = POIRankingEngine()

    logger.info(f">> [초기화 완료] 총 소요시간: {time.time() - start_time:.2f}초. 서비스 준비 완료.")
    yield
    logger.info(">> [서버 종료] 메모리 리소스를 정리합니다.")
    app_state.clear()


app = FastAPI(
    title="GoldenStep AI Simulation API",
    description="보행 네트워크 기반 치매 실종자 도달 권역 및 우선 확인 거점 TOP 3 추천 엔진",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 미들웨어 (프론트엔드/백엔드 로컬 및 배포 도메인 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["System"])
def health_check():
    """서버 상태 및 엔진 적재 여부 확인 (컨테이너/ALB 헬스체크 표준)"""
    is_ready = bool(
        app_state.get("iso_engine") is not None and 
        app_state.get("poi_engine") is not None and 
        app_state.get("simulator") is not None
    )
    
    if not is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "INITIALIZING",
                "message": "시뮬레이션 엔진 및 도로망 데이터 적재 중...",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "UP",
            "message": "GoldenStep AI Engine Ready",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


@app.post(
    "/api/v1/simulation/search",
    response_model=SearchSimulationResponse,
    status_code=status.HTTP_200_OK,
    tags=["Simulation"]
)
def predict_simulation(request: SearchSimulationRequest):
    """
    [Core MVP 실연동 API]
    실종자의 마지막 위치와 경과 시간을 입력받아,
    1. 보행 네트워크 기반 도달 권역(A1/A2 Isochrone Polygon) 산출
    2. 몬테카를로 에이전트 2,000명 시뮬레이션 및 실제 이동 궤적(A3) 도로망 추출
    3. 통과 궤적 밀도와 공간 인접도를 결합하여 권역 내부 유력 거점 TOP 3 선별
    4. 거점 기준 반경별(100m, 300m, 500m) 존재 확률 산출
    """
    start_perf = time.perf_counter()
    
    iso_engine: IsochroneEngine = app_state.get("iso_engine")
    poi_engine: POIRankingEngine = app_state.get("poi_engine")
    simulator: MultiTypeRandomWalkSimulator = app_state.get("simulator")

    if iso_engine is None or poi_engine is None or simulator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="시뮬레이션 엔진이 아직 메모리에 적재되지 않았습니다."
        )

    try:
        mp = request.missing_person
        lat = mp.last_seen_location.lat
        lon = mp.last_seen_location.lon
        raw_elapsed_h = mp.elapsed_hours

        # [성능 최적화] 치매 환자 도보 이동 한계(Koester ISRID 통계: 75% 3.8시간)를 반영한 시뮬레이션 계산 상한
        sim_elapsed_h = min(raw_elapsed_h, 4.0)

        base_speed = settings.BASE_WALK_VELOCITY_KMH
        agent_count = request.parameters.num_agents if request.parameters else settings.NUM_SIMULATION_AGENTS

        # -------------------------------------------------------------
        # 1. 보행 도로망 도달 권역(A1/A2 Isochrone) 계산
        # -------------------------------------------------------------
        iso_feature = iso_engine.calculate_isochrone(
            center_lat=lat, 
            center_lon=lon, 
            elapsed_hours=raw_elapsed_h,
            base_speed_kmh=base_speed,
            exact_union=False
        )
        boundary_polygon = shape(iso_feature["geometry"])
        iso_props = iso_feature["properties"]

        # -------------------------------------------------------------
        # 2. 몬테카를로 시뮬레이션 선행 실행 (에이전트 통과 도로망 A3 및 최종 정지점 추출)
        # -------------------------------------------------------------
        sim_result = simulator.simulate(
            center_lat=lat,
            center_lon=lon,
            num_agents=agent_count,
            max_hours=sim_elapsed_h,
            base_speed=base_speed,
            beta_dp=1.0,
            beta_cost=1.0,
            beta_wid=1.0,
            return_polygon=False
        )
        edge_visit_counts = sim_result.get("edge_visit_counts", {})
        stopped_points = sim_result.get("stopped_points", [])

        # -------------------------------------------------------------
        # 3. 에이전트 궤적 및 정지점 기반 POI 랭킹 및 존재 확률 연산
        # -------------------------------------------------------------
        raw_pois = poi_engine.rank_points_of_interest(
            boundary_polygon=boundary_polygon,
            origin_lat=lat,
            origin_lon=lon,
            edge_visit_counts=edge_visit_counts,
            terminal_edge_counts=sim_result.get("terminal_edge_counts", {}),
            stopped_points=stopped_points,
            graph=simulator.graph,
            top_k=3
        )

        # -------------------------------------------------------------
        # 4. PriorityPoint DTO 조립 (백엔드 txt 규격: confidence, deep_links 제외)
        # -------------------------------------------------------------
        priority_point_list = []
        for p in raw_pois:
            pa_raw = p.get("probability_analysis")
            pa_dto = None
            if pa_raw:
                pa_dto = ProbabilityAnalysis(
                    relative_priority_prob=pa_raw["relative_priority_prob"],
                    range_probabilities=RangeProbabilities(
                        within_100m=pa_raw["range_probabilities"]["within_100m"],
                        within_300m=pa_raw["range_probabilities"]["within_300m"],
                        within_500m=pa_raw["range_probabilities"]["within_500m"]
                    )
                )

            priority_point_list.append(
                PriorityPoint(
                    rank=p["rank"],
                    poi_id=p["poi_id"],
                    name=p["name"],
                    category=p["category"],
                    location=POILocation(lat=p["location"]["lat"], lon=p["location"]["lon"]),
                    distance_m=p.get("distance_m"),
                    score=p["score"],
                    probability_analysis=pa_dto,
                    recommendation_reason=p["recommendation_reason"]
                )
            )

        # -------------------------------------------------------------
        # 5. 신뢰도 등급 산정
        # -------------------------------------------------------------
        if raw_elapsed_h <= 1.5:
            reliability = "STABLE"
            warning_msg = None
        elif raw_elapsed_h <= 3.0:
            reliability = "CAUTION"
            warning_msg = "실종 후 1.5시간 이상 경과하여 이동 반경이 넓어졌습니다. 대중교통 이용 가능성에 유의하세요."
        else:
            reliability = "REFERENCE"
            warning_msg = "실종 후 3시간 이상 경과하여 보행 예측 신뢰도가 낮습니다. 112 긴급 신고를 병행하십시오."

        # -------------------------------------------------------------
        # 6. 요약 통계(Summary) 생성 (백엔드 txt 규격: base_velocity_kmh, max_distance_m 제외)
        # -------------------------------------------------------------
        summary = SimulationSummary(
            person_type=mp.person_type,
            elapsed_hours=raw_elapsed_h,
            reliability_status=reliability,
            reliability_warning=warning_msg,
            area_reduction_rate=float(str(iso_props["area_reduction_rate_a2"]).replace("%", ""))
        )

        # -------------------------------------------------------------
        # 7. FeatureCollection 래핑
        # -------------------------------------------------------------
        boundary_collection = BoundaryZoneCollection(
            features=[
                BoundaryFeature(
                    properties=iso_props,
                    geometry=GeoJSONGeometry(
                        type=iso_feature["geometry"]["type"],
                        coordinates=iso_feature["geometry"]["coordinates"]
                    )
                )
            ]
        )

        exec_time = round(time.perf_counter() - start_perf, 3)

        # -------------------------------------------------------------
        # 8. 최종 반환 (백엔드 txt 규격: high_probability_edges 제외)
        # -------------------------------------------------------------
        return SearchSimulationResponse(
            status="SUCCESS",
            request_id=request.request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            execution_time_sec=exec_time,
            summary=summary,
            boundary_zone=boundary_collection,
            priority_points=priority_point_list
        )

    except Exception as e:
        logger.error(f"시뮬레이션 연산 중 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"추론 엔진 연산 실패: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.SERVER_HOST, port=settings.SERVER_PORT, reload=settings.DEBUG)