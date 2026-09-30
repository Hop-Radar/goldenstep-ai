"""
main.py
- GoldenStep Core AI Simulation API Server
- 엔드포인트: POST /api/v1/simulation/search (BE 연동 표준 규격)
- FastAPI Lifespan을 통한 도로망(Graph) 및 POI R-Tree 인메모리 사전 적재 (SLA < 0.5초)
- [수정] MultiTypeRandomWalkSimulator 연동 및 high_probability_edges 응답 주입 완료
"""

import time
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
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
    DeepLinks,
    HighProbabilityEdgesCollection,
    HighProbabilityEdgeFeature
)
from core.graph_loader import WalkGraphManager
from core.isochrone_engine import IsochroneEngine
from core.agent_simulator import MultiTypeRandomWalkSimulator
from core.poi_ranking_engine import POIRankingEngine
from core.profiles.profile_factory import ProfileFactory
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
    """서버 상태 및 엔진 적재 여부 확인"""
    is_ready = "iso_engine" in app_state and "poi_engine" in app_state and "simulator" in app_state
    return {
        "status": "UP" if is_ready else "INITIALIZING",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


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
    1. 보행 네트워크 기반 도달 권역(Isochrone Polygon) 산출
    2. R-Tree 공간 필터링 및 행동역학 가중치 기반 우선 거점 TOP 3 선별
    3. 네이버 지도 도보 길찾기 외부 딥링크 생성
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
        elapsed_h = mp.elapsed_hours

        # 도메인별 프로필 로드
        profile = ProfileFactory.get_profile(mp.person_type)
        base_speed = profile.get("base_speed_kmh", 2.40)
        fatigue = profile.get("fatigue_lambda", 0.15)
        cat_weights = profile.get("category_weights", {})

        # 1. 보행 도로망 도달 권역(Isochrone Feature) 계산
        if mp.person_type == "DEMENTIA":
            profile_severe = ProfileFactory.get_profile("DEMENTIA_SEVERE")
            profile_mild = ProfileFactory.get_profile("DEMENTIA_MILD")
            
            # Run A (Severe): No turn penalty (Local Random Wandering)
            iso_a = iso_engine.calculate_isochrone(
                center_lat=lat, center_lon=lon, elapsed_hours=elapsed_h,
                base_speed_kmh=profile_severe.get("base_speed_kmh", 2.40),
                fatigue_lambda=profile_severe.get("fatigue_lambda", 0.30),
                turn_penalty_factor=0.0
            )
            # Run B (Mild): With turn penalty (Goal-directed Straight Wandering)
            iso_b = iso_engine.calculate_isochrone(
                center_lat=lat, center_lon=lon, elapsed_hours=elapsed_h,
                base_speed_kmh=profile_mild.get("base_speed_kmh", 2.40),
                fatigue_lambda=profile_mild.get("fatigue_lambda", 0.10),
                turn_penalty_factor=1.5
            )
            
            from shapely.geometry import mapping
            zone_a = shape(iso_a["geometry"])
            zone_b = shape(iso_b["geometry"])
            
            # 합집합(Union)으로 전체 앙상블 바운더리 생성
            boundary_polygon = zone_a.union(zone_b)
            iso_feature = iso_b  # 속성(Properties)은 범위가 넓은 경증 기준 적용
            iso_feature["geometry"] = mapping(boundary_polygon)
        else:
            iso_feature = iso_engine.calculate_isochrone(
                center_lat=lat, center_lon=lon, elapsed_hours=elapsed_h,
                base_speed_kmh=base_speed, fatigue_lambda=fatigue
            )
            boundary_polygon = shape(iso_feature["geometry"])
            zone_a = None
            zone_b = None

        iso_geom_dict = iso_feature["geometry"]
        iso_props = iso_feature["properties"]

        # =========================================================================
        # [변경 지점] 2. 도달 권역 내부 POI 공간 필터링 및 TOP 3 거점 선행 산출
        # =========================================================================
        raw_pois = poi_engine.rank_points_of_interest(
            boundary_polygon=boundary_polygon,
            origin_lat=lat,
            origin_lon=lon,
            top_k=3,
            category_weights=cat_weights,
            zone_a_polygon=zone_a,
            zone_b_polygon=zone_b
        )

        # =========================================================================
        # [변경 지점] 3. 몬테카를로 시뮬레이션 및 우선순위 거점 중심 300m 회랑 추출 연동
        # =========================================================================
        agent_count = request.parameters.num_agents if request.parameters else settings.NUM_SIMULATION_AGENTS
        sim_result = simulator.simulate(
            center_lat=lat,
            center_lon=lon,
            num_agents=agent_count,
            max_hours=elapsed_h,
            base_speed=base_speed
        )
        
        # [변경 지점] 거점 중심 300m 완충 구역 발걸음 및 유입 경로 선별 호출
        raw_edges_geojson = simulator.extract_poi_corridors(
            agents=sim_result.get("agents", []),
            top_pois=raw_pois,
            radius_m=300.0
        )
        
        # 4. HighProbabilityEdgesCollection DTO 조립
        edge_feature_list = []
        for feat in raw_edges_geojson.get("features", []):
            edge_feature_list.append(
                HighProbabilityEdgeFeature(
                    properties=feat.get("properties", {}),
                    geometry=GeoJSONGeometry(
                        type=feat["geometry"]["type"],
                        coordinates=feat["geometry"]["coordinates"]
                    )
                )
            )
        edges_collection = HighProbabilityEdgesCollection(
            type="FeatureCollection",
            features=edge_feature_list
        )

        # 5. PriorityPoint DTO 조립
        priority_point_list = [
            PriorityPoint(
                rank=p["rank"],
                poi_id=p["poi_id"],
                name=p["name"],
                category=p["category"],
                location=POILocation(lat=p["location"]["lat"], lon=p["location"]["lon"]),
                distance_m=p.get("distance_m"),
                score=p["score"],
                recommendation_reason=p["recommendation_reason"],
                deep_links=DeepLinks(
                    naver_map=p["deep_links"]["naver_map"],
                    kakao_map=p["deep_links"]["kakao_map"]
                )
            )
            for p in raw_pois
        ]

        # 6. 신뢰도 등급(Reliability Status) 산정 (기획서 PB-06)
        if elapsed_h <= 1.5:
            reliability = "STABLE"
            warning_msg = None
        elif elapsed_h <= 3.0:
            reliability = "CAUTION"
            warning_msg = "실종 후 1.5시간 이상 경과하여 이동 반경이 넓어졌습니다. 대중교통 이용 가능성에 유의하세요."
        else:
            reliability = "REFERENCE"
            warning_msg = "실종 후 3시간 이상 경과하여 보행 예측 신뢰도가 낮습니다. 112 긴급 신고를 병행하십시오."

        # 7. 요약 통계(Summary) 생성
        summary = SimulationSummary(
            person_type=mp.person_type,
            base_velocity_kmh=base_speed,
            elapsed_hours=elapsed_h,
            max_distance_m=iso_props["cutoff_distance_m"],
            reliability_status=reliability,
            reliability_warning=warning_msg,
            area_reduction_rate=round(iso_props["arr_raw"] * 100.0, 1)
        )

        # 8. FeatureCollection 래핑
        boundary_collection = BoundaryZoneCollection(
            features=[
                BoundaryFeature(
                    properties=iso_props,
                    geometry=GeoJSONGeometry(
                        type=iso_geom_dict["type"],
                        coordinates=iso_geom_dict["coordinates"]
                    )
                )
            ]
        )

        exec_time = round(time.perf_counter() - start_perf, 3)

        return SearchSimulationResponse(
            status="SUCCESS",
            request_id=request.request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            execution_time_sec=exec_time,
            summary=summary,
            boundary_zone=boundary_collection,
            priority_points=priority_point_list,
            high_probability_edges=edges_collection
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