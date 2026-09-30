"""
core/poi_ranking_engine.py
- GoldenStep Core Engine: 도달 권역 내부 POI 추출 및 행동역학 다기준 랭킹 엔진
- 데이터 소스: data/pois/seoul_public_pois.geojson (2,934건)
- R-Tree(STRtree) 기반 초고속 공간 필터링 (0.01초 이내)
- 기획서 v1.1(PB-03, PB-05) TOP 1~3 순위 선정 및 네이버 지도 길찾기 딥링크 생성
- [수정] 100m 이내 인접 중복 POI 배제 (다양성 필터)
"""

import os
import json
import urllib.parse
import logging
import pandas as pd
from typing import List, Dict, Any, Tuple
from shapely.geometry import Point, Polygon, shape
from shapely.strtree import STRtree
from core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GoldenStep-POIRanker")

# 도메인 확장에 따라 프로필(JSON)에서 가중치를 동적으로 로드합니다.
# 기본 가중치는 1.0으로 일괄 적용되며, rank_points_of_interest에 인자로 전달됩니다.


class POIRankingEngine:
    def __init__(self, 
                 geojson_path: str = None,
                 parquet_path: str = None):
        """
        GeoJSON (공공시설) 및 Parquet (건축물대장 69만건) POI 데이터를 통합 적재하고
        Shapely STRtree 공간 인덱스를 초기화합니다.
        """
        self.geojson_path = geojson_path or settings.PUBLIC_POI_DATA_PATH
        self.parquet_path = parquet_path or settings.BUILDING_POI_DATA_PATH
        self.poi_records: List[Dict[str, Any]] = []
        self.geometries: List[Point] = []
        
        self._load_and_index_pois()

    def _load_and_index_pois(self):
        """GeoJSON 및 Parquet을 읽어 R-Tree 공간 인덱스(STRtree) 통합 구축"""
        # 1. GeoJSON 로드 (공공/교통/자연)
        if os.path.exists(self.geojson_path):
            with open(self.geojson_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            features = data.get("features", [])
            for feat in features:
                props = feat.get("properties", {})
                geom = feat.get("geometry", {})
                coords = geom.get("coordinates", [])
                
                if len(coords) >= 2:
                    lon, lat = coords[0], coords[1]
                    pt = Point(lon, lat)
                    self.geometries.append(pt)
                    self.poi_records.append({
                        "poi_id": props.get("poi_id", ""),
                        "name": props.get("name", "지정 거점"),
                        "category": props.get("category", "OTHER"),
                        "base_weight": props.get("base_weight", 1.0),
                        "has_shelter": props.get("has_shelter", False),
                        "has_lit": props.get("has_lit", False),
                        "lat": lat,
                        "lon": lon
                    })
            logger.info(f">> GeoJSON 공공시설 로드 완료: {len(features):,}건")
        else:
            logger.warning(f"GeoJSON 데이터를 찾을 수 없습니다: {self.geojson_path}")

        # 2. Parquet 로드 (건축물대장 69만건)
        if os.path.exists(self.parquet_path):
            try:
                df = pd.read_parquet(self.parquet_path)
                # DataFrame iteration
                for row in df.itertuples():
                    pt = Point(row.lon, row.lat)
                    self.geometries.append(pt)
                    self.poi_records.append({
                        "poi_id": row.poi_id,
                        "name": row.name,
                        "category": row.category,
                        "base_weight": row.base_weight,
                        "has_shelter": True, # 건물은 기본적으로 쉘터 역할을 수행 가능
                        "has_lit": True,
                        "lat": row.lat,
                        "lon": row.lon
                    })
                logger.info(f">> Parquet 건축물대장 로드 완료: {len(df):,}건")
            except Exception as e:
                logger.error(f"Parquet 로드 실패: {e}")
        else:
            logger.warning(f"Parquet 건축물대장 데이터를 찾을 수 없습니다: {self.parquet_path}")

        if not self.geometries:
            logger.warning("POI 데이터가 하나도 없습니다. 빈 인덱스로 초기화합니다.")
            self.tree = STRtree([])
            return

        # 3. Shapely STRtree 생성 (공간 질의 속도 수 밀리초 보장)
        self.tree = STRtree(self.geometries)
        logger.info(f">> POI R-Tree 통합 인덱스 구축 완료: 총 {len(self.poi_records):,}개 거점 적재됨.")

    def _generate_recommendation_reason(self, category: str, name: str, dist_m: float) -> str:
        """수색대 및 현장 요원을 위한 행동역학 기반 직관적 추천 사유 생성"""
        dist_str = f"약 {int(dist_m)}m" if dist_m else "도보권"
        
        # 1. 주거 및 대단지 구조물 (도심 최다 발견 지형: 35%)
        if category in ["APARTMENT", "HOME", "VILLA", "RESIDENTIAL_AREA"]:
            return f"실종 지점 {dist_str} 거리의 주거 구역으로, 과거 거주지 착각 진입 및 단지 내 필로티·계단 등 실내 공간 체류 가능성 높음"
        
        # 2. 대중교통 및 결절점
        elif category in ["BUS_STOP", "SUBWAY_STATION"]:
            return f"주요 이동축 결절점({dist_str})에 위치하며, 쉼터 및 벤치가 있어 보행 피로 시 머무르거나 이동하기 위해 대기할 가능성 높음"
        
        # 3. 상업 시설 (조명 유인 및 목격 다발)
        elif category in ["STORE", "MARKET", "COMMERCIAL_BUILDING"]:
            return f"보행로 인접 상업 시설({dist_str})로, 조명에 의한 시선 유인 및 실내 진입 가능성 높음"
        
        # 4. 공원 및 휴식 공간
        elif category in ["PARK", "REST_SHELTER", "FOREST", "BUSH"]:
            return f"접근 가능한 쉼터 구역({dist_str})으로, 신체 피로 누적으로 인한 공원 및 휴식 공간의 벤치, 숲길 등 체류 가능성 높음"
        
        # 5. 병원 및 공공 보호 거점
        elif category in ["HOSPITAL"]:
            return f"보호 및 치료 시설({dist_str})로, 친숙한 시설 지향 또는 시민·관계자에 의한 조기 발견 및 조치되어 있을 가능성 높음"
        
        # 6. 도로 및 일반 보행로
        elif category in ["ROAD", "INTERSECTION"]:
            return f"이동 동선 상 주요 길목({dist_str})으로, 배회 및 통과 확률이 높은 길목임"
        
        # 7. 기본 폴백
        return f"도달 가능한 위치({dist_str})에 위치한 거점임"
    
    def rank_points_of_interest(
        self,
        boundary_polygon: Polygon,
        origin_lat: float,
        origin_lon: float,
        top_k: int = 3,
        category_weights: Dict[str, float] = None,
        zone_a_polygon: Polygon = None,
        zone_b_polygon: Polygon = None
    ) -> List[Dict[str, Any]]:
        """
        Isochrone 도달 권역 다각형 내부의 POI를 공간 필터링하고
        행동역학 다기준 점수를 채점하여 상위 TOP K 거점을 반환합니다.
        """
        if category_weights is None:
            # Safe182 실측 데이터 기반의 '약한 가중치(Weak Weights)' 적용
            # 통계: HOME(1.6만) > APARTMENT(4천) > ROAD(3천) > RESIDENTIAL(1.7천) > HOSPITAL(1.5천) > VILLA(1천)
            category_weights = {
                "HOME": 1.8,
                "APARTMENT": 1.7,
                "ROAD": 1.6,
                "INTERSECTION": 1.6,
                "RESIDENTIAL_AREA": 1.5,
                "VILLA": 1.5,
                "HOSPITAL": 1.4,
                "COMMERCIAL_BUILDING": 1.4,
                "MARKET": 1.3,
                "SUBWAY_STATION": 1.3,
                "BUS_STOP": 1.3,
                "PARK": 1.2,
                "STORE": 1.2,
                "OFFICE": 1.1,
                "REST_SHELTER": 1.1,
                "MOUNTAIN": 1.1,
                "FOREST": 1.1,
                "BUSH": 1.0,
                "RIVER": 1.0,
                "OTHER": 1.0
            }

        if not self.geometries or boundary_polygon.is_empty:
            return []

        # 1. R-Tree Bounding Box 1차 초고속 후보 추출
        candidate_indices = self.tree.query(boundary_polygon)
        matched_pois = []

        for idx in candidate_indices:
            pt = self.geometries[idx]
            # 2. 정밀 공간 포함 검사 (Point-in-Polygon)
            if boundary_polygon.contains(pt):
                row = self.poi_records[idx]

                # 3. 실종 지점으로부터의 물리적 도보 이동 거리 계산 (m)
                d_lat = (row['lat'] - origin_lat) * 111000.0
                d_lon = (row['lon'] - origin_lon) * 88800.0
                dist_m = (d_lat**2 + d_lon**2)**0.5

                # 4. 행동역학 가중치 산출
                cat = row['category']
                weight = category_weights.get(cat, row.get('base_weight', 1.0))
                
                # Zone Priority 앙상블 적용 (A ∩ B 1순위 처리)
                if zone_a_polygon and zone_b_polygon:
                    in_a = zone_a_polygon.contains(pt)
                    in_b = zone_b_polygon.contains(pt)
                    if in_a and in_b:
                        weight *= 3.0  # 1순위 (Red Zone)
                    elif in_a:
                        weight *= 1.5  # 2순위 (Yellow Zone)
                    elif in_b:
                        weight *= 1.0  # 3순위 (Blue Zone)
                
                # 가중치 추가 보정 (쉘터/조명 시설 가점)
                if row.get('has_shelter'):
                    weight += 0.1
                if row.get('has_lit'):
                    weight += 0.1

                # 5. 복합 점수 산출식: (시설 행동 유인 50%) + (거리 접근성 감쇠 50%)
                distance_score = 1.0 / (1.0 + (dist_m / 1000.0))
                raw_score = (weight / 2.2) * 0.5 + distance_score * 0.5
                score = round(min(0.99, raw_score), 3)

                # 6. 네이버 지도 / 카카오맵 도보 길찾기 외부 딥링크 생성 (기획서 PB-05)
                name_enc = urllib.parse.quote(str(row['name']))
                naver_map_link = (
                    f"nmap://route/walk?dlat={row['lat']}&dlng={row['lon']}"
                    f"&dname={name_enc}&appname=com.goldenstep.app"
                )
                kakao_map_link = f"kakaomap://route?ep={row['lat']},{row['lon']}&by=FOOT"

                matched_pois.append({
                    "poi_id": row['poi_id'],
                    "name": row['name'],
                    "category": cat,
                    "location": {
                        "lat": row['lat'],
                        "lon": row['lon']
                    },
                    "distance_m": round(dist_m, 1),
                    "score": score,
                    "raw_score": raw_score,
                    "deep_links": {
                        "naver_map": naver_map_link,
                        "kakao_map": kakao_map_link
                    }
                })

        # 7. 점수 기준 내림차순 정렬 후 상위 TOP K 추출 (내부 랭킹은 캡핑되지 않은 raw_score 기준)
        matched_pois.sort(key=lambda x: x['raw_score'], reverse=True)
        top_pois = matched_pois[:top_k]
        # 100m 이내 인접 동일 카테고리 POI 제거 (다양성 NMS 필터)
        deduped_pois = []
        for p in matched_pois:
            is_dup = False
            for accepted in deduped_pois:
                dy = (p["location"]["lat"] - accepted["location"]["lat"]) * 111000.0
                dx = (p["location"]["lon"] - accepted["location"]["lon"]) * 88800.0
                sep_dist = (dx**2 + dy**2)**0.5
                if sep_dist < 100.0 and p["category"] == accepted["category"]:
                    is_dup = True
                    break
            if not is_dup:
                deduped_pois.append(p)
            if len(deduped_pois) >= top_k:
                break

        # 8. 랭크 순위 부여 및 추천 사유 생성
        for rank, p in enumerate(deduped_pois, start=1):
            p['rank'] = rank
            p['recommendation_reason'] = self._generate_recommendation_reason(
                category=p['category'],
                name=p['name'],
                dist_m=p.get('distance_m', 0.0)
            )

        return deduped_pois