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
    elapsed_hours: float
    reliability_status: str
    reliability_warning: Optional[str] = None
    area_reduction_rate: float

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

class RangeProbabilities(BaseModel):
    within_100m: float
    within_300m: float
    within_500m: float

class ProbabilityAnalysis(BaseModel):
    relative_priority_prob: float
    range_probabilities: RangeProbabilities

class PriorityPoint(BaseModel):
    rank: int
    poi_id: str
    name: str
    category: str
    location: POILocation
    distance_m: Optional[float] = None
    score: float
    probability_analysis: Optional[ProbabilityAnalysis] = None
    recommendation_reason: str

class SearchSimulationResponse(BaseModel):
    status: str = "SUCCESS"
    request_id: str
    timestamp: str
    execution_time_sec: float
    summary: SimulationSummary
    boundary_zone: BoundaryZoneCollection
    priority_points: List[PriorityPoint]