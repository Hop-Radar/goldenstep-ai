"""
api/schemas.py
- GoldenStep API Request/Response Pydantic 스키마 정의
- 기획서 명세 및 mock_search_request.json / mock_search_response.json 100% 호환
- [확장] PriorityPoint에 거점 중심 반경별 존재 확률(100m, 300m, 500m) 및 통계적 신뢰도 스키마 추가
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from core.config import settings


# ==================== [요청 Request Schemas] ====================

class LastSeenLocation(BaseModel):
    lat: float = Field(..., description="마지막 확인 위치 위도", example=37.5398)
    lon: float = Field(..., description="마지막 확인 위치 경도", example=126.9479)
    address: Optional[str] = Field(None, description="주소 명칭", example="서울특별시 마포구 도화동 사우나 부근")


class MissingPersonInfo(BaseModel):
    person_type: str = Field(default="DEMENTIA", description="대상자 유형 (기본: DEMENTIA)")
    last_seen_location: LastSeenLocation
    last_seen_time: Optional[str] = Field(None, description="마지막 확인 시각 (ISO8601)")
    elapsed_hours: float = Field(..., gt=0.0, description="실종 경과 시간 (hours)", example=1.0)


class SimulationParameters(BaseModel):
    num_agents: Optional[int] = Field(default=settings.NUM_SIMULATION_AGENTS, description="가상 에이전트 수")
    search_radius_m: Optional[float] = Field(default=settings.DEFAULT_SEARCH_RADIUS_M, description="검색 네트워크 반경(m)")
    fallback_level: Optional[str] = Field(default="PLAN_A", description="폴백 전략 단계")


class SearchSimulationRequest(BaseModel):
    request_id: str = Field(..., description="요청 고유 UUID", example="req_20260928_001")
    missing_person: MissingPersonInfo
    parameters: Optional[SimulationParameters] = Field(default_factory=SimulationParameters)


# ==================== [응답 Response Schemas] ====================

class SimulationSummary(BaseModel):
    person_type: str
    base_velocity_kmh: float
    elapsed_hours: float
    max_distance_m: float
    reliability_status: str = Field(..., description="STABLE, CAUTION, REFERENCE 중 하나")
    reliability_warning: Optional[str] = None
    area_reduction_rate: float = Field(..., description="단순 동심원 대비 축소율 (%)")


class GeoJSONGeometry(BaseModel):
    type: str
    coordinates: List[Any]


class BoundaryFeature(BaseModel):
    type: str = "Feature"
    properties: Dict[str, Any]
    geometry: GeoJSONGeometry


class BoundaryZoneCollection(BaseModel):
    type: str = "FeatureCollection"
    features: List[BoundaryFeature]


class POILocation(BaseModel):
    lat: float
    lon: float


class DeepLinks(BaseModel):
    naver_map: str
    kakao_map: Optional[str] = None


# ==================== [신규 확장: 확률 및 신뢰도 분석 스키마] ====================

class RangeProbabilities(BaseModel):
    within_100m: float = Field(..., description="거점 반경 100m 이내 실종자 체류 확률 (0.0~1.0)")
    within_300m: float = Field(..., description="거점 반경 300m 이내 실종자 체류 확률 (0.0~1.0)")
    within_500m: float = Field(..., description="거점 반경 500m 이내 실종자 체류 확률 (0.0~1.0)")


class ConfidenceStats(BaseModel):
    confidence_level: float = Field(default=0.95, description="모비율 추정 통계적 신뢰수준 (기본 0.95)")
    margin_of_error_pct: float = Field(..., description="300m 기준 95% 신뢰구간 표본 오차 범위 (±%)")
    dispersion_grade: str = Field(..., description="에이전트 공간 수렴도 (HIGH, MODERATE, LOW)")
    sample_size: int = Field(default=settings.NUM_SIMULATION_AGENTS, description="시뮬레이션 가상 에이전트 표본 수")


class ProbabilityAnalysis(BaseModel):
    relative_priority_prob: float = Field(..., description="Top-3 거점 간 상대적 우선 선택 확률")
    range_probabilities: RangeProbabilities
    confidence: ConfidenceStats


class PriorityPoint(BaseModel):
    rank: int
    poi_id: str
    name: str
    category: str
    location: POILocation
    distance_m: Optional[float] = Field(None, description="실종 지점 기준 도보 거리(m)")
    score: float
    probability_analysis: Optional[ProbabilityAnalysis] = Field(None, description="거점 기준 반경별 존재 확률 및 신뢰도")
    recommendation_reason: str
    deep_links: DeepLinks


class HighProbabilityEdgeFeature(BaseModel):
    type: str = "Feature"
    properties: Dict[str, Any]
    geometry: GeoJSONGeometry


class HighProbabilityEdgesCollection(BaseModel):
    type: str = "FeatureCollection"
    features: List[HighProbabilityEdgeFeature]


class SearchSimulationResponse(BaseModel):
    status: str = "SUCCESS"
    request_id: str
    timestamp: str
    execution_time_sec: float
    summary: SimulationSummary
    boundary_zone: BoundaryZoneCollection
    priority_points: List[PriorityPoint]
    high_probability_edges: Optional[HighProbabilityEdgesCollection] = None