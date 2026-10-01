"""
core/poi_ranking_engine.py
- GoldenStep V2.0 Core: 에이전트 이동 궤적(A3) 연동 결정론적 우선 확인 거점(Top 3) 랭킹 엔진
- [수정] 단순 거리 역수 감점 1/(1+d/1000) 완전 삭제
- [수정] 에이전트 최종 체류 도로(Terminal Edges) 및 주요 통과 도로 인접도 결합 채점
- [수정] 카테고리 무관 거점 간 최소 150m 물리적 이격 NMS 다양성 필터 적용
"""

import os
import json
import urllib.parse
import logging
import pandas as pd
from typing import List, Dict, Any, Tuple
from shapely.geometry import Point, Polygon
from shapely.strtree import STRtree
from core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GoldenStep-POIRanker")


class POIRankingEngine:
    def __init__(self, 
                 geojson_path: str = None,
                 parquet_path: str = None):
        self.geojson_path = geojson_path or settings.PUBLIC_POI_DATA_PATH
        self.parquet_path = parquet_path or settings.BUILDING_POI_DATA_PATH
        self.poi_records: List[Dict[str, Any]] = []
        self.geometries: List[Point] = []
        
        self._load_and_index_pois()

    def _load_and_index_pois(self):
        if os.path.exists(self.geojson_path):
            try:
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
                            "lat": lat,
                            "lon": lon
                        })
                logger.info(f">> GeoJSON 공공시설 로드 완료: {len(features):,}건")
            except Exception as e:
                logger.error(f"GeoJSON 로드 실패: {e}")

        if os.path.exists(self.parquet_path):
            try:
                df = pd.read_parquet(self.parquet_path)
                for row in df.itertuples():
                    pt = Point(row.lon, row.lat)
                    self.geometries.append(pt)
                    self.poi_records.append({
                        "poi_id": getattr(row, "poi_id", ""),
                        "name": getattr(row, "name", "건물"),
                        "category": getattr(row, "category", "BUILDING"),
                        "lat": row.lat,
                        "lon": row.lon
                    })
                logger.info(f">> Parquet 건축물대장 로드 완료: {len(df):,}건")
            except Exception as e:
                logger.error(f"Parquet 로드 실패: {e}")

        if not self.geometries:
            self.tree = STRtree([])
            return

        self.tree = STRtree(self.geometries)
        logger.info(f">> POI R-Tree 통합 인덱스 구축 완료: 총 {len(self.poi_records):,}개 거점 적재됨.")

    def _generate_recommendation_reason(self, category: str, dist_m: float) -> str:
        dist_str = f"약 {int(dist_m)}m" if dist_m else "도보권"
        if category in ["APARTMENT", "HOME", "VILLA", "RESIDENTIAL_AREA", "BUILDING"]:
            return f"실종 지점 {dist_str} 거리의 주거/건물 구역으로, 에이전트 이동 집중 경로에 위치하여 체류 확인 필요"
        elif category in ["BUS_STOP", "SUBWAY_STATION"]:
            return f"주요 이동축 결절점({dist_str})으로, 보행 피로 시 대기 가능성 높음"
        elif category in ["STORE", "MARKET", "COMMERCIAL_BUILDING"]:
            return f"보행로 인접 상가({dist_str})로, 실내 진입 확인 필요"
        elif category in ["PARK", "REST_SHELTER", "FOREST"]:
            return f"개방형 쉼터 구역({dist_str})으로, 보행 경로 상 휴식 공간"
        return f"이동 권역({dist_str}) 내 주요 확인 거점"

    def rank_points_of_interest(
        self,
        boundary_polygon: Polygon,
        origin_lat: float,
        origin_lon: float,
        edge_visit_counts: Dict[Tuple[int, int], int] = None,
        terminal_edge_counts: Dict[Tuple[int, int], int] = None,
        graph = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        에이전트 최종 체류 도로(Terminal) 및 주요 이동 통과 도로에 접면한 POI를 선별
        """
        if not self.geometries or boundary_polygon.is_empty:
            return []

        candidate_indices = self.tree.query(boundary_polygon)
        if len(candidate_indices) == 0:
            return []

        # 1. 에이전트 가중치 중심점 구축 (체류 도로 70% + 통과 도로 30%)
        weighted_edge_points = []
        if graph is not None:
            # 최종 체류 도로 반영
            if terminal_edge_counts:
                max_term = max(terminal_edge_counts.values()) if terminal_edge_counts else 1
                for (u, v), count in terminal_edge_counts.items():
                    if u in graph.nodes and v in graph.nodes:
                        mid_x = (graph.nodes[u]['x'] + graph.nodes[v]['x']) / 2.0
                        mid_y = (graph.nodes[u]['y'] + graph.nodes[v]['y']) / 2.0
                        w = (count / max_term) * 1.0  # 최고 가중치 부여
                        weighted_edge_points.append((mid_y, mid_x, w))

            # 주요 통과 도로 반영
            if edge_visit_counts:
                max_visit = max(edge_visit_counts.values()) if edge_visit_counts else 1
                for (u, v), count in edge_visit_counts.items():
                    if u in graph.nodes and v in graph.nodes:
                        mid_x = (graph.nodes[u]['x'] + graph.nodes[v]['x']) / 2.0
                        mid_y = (graph.nodes[u]['y'] + graph.nodes[v]['y']) / 2.0
                        w = (count / max_visit) * 0.5
                        weighted_edge_points.append((mid_y, mid_x, w))

        matched_pois = []

        for idx in candidate_indices:
            pt = self.geometries[idx]
            if boundary_polygon.contains(pt):
                row = self.poi_records[idx]

                d_lat = (row['lat'] - origin_lat) * 111000.0
                d_lon = (row['lon'] - origin_lon) * 88800.0
                dist_m = (d_lat**2 + d_lon**2)**0.5

                # 에이전트 체류 및 주행 도로 접면도 (Track Offset 50m 이내)
                road_affinity_score = 0.0
                if weighted_edge_points:
                    min_dist = float('inf')
                    best_w = 0.0
                    for c_lat, c_lon, w in weighted_edge_points[:200]:
                        dy = (row['lat'] - c_lat) * 111000.0
                        dx = (row['lon'] - c_lon) * 88800.0
                        d = (dx**2 + dy**2)**0.5
                        if d < min_dist:
                            min_dist = d
                            best_w = w

                    if min_dist <= 50.0:
                        road_affinity_score = best_w * (1.0 - (min_dist / 50.0))

                name_enc = urllib.parse.quote(str(row['name']))
                naver_map_link = (
                    f"nmap://route/walk?dlat={row['lat']}&dlng={row['lon']}"
                    f"&dname={name_enc}&appname=com.goldenstep.app"
                )
                kakao_map_link = f"kakaomap://route?ep={row['lat']},{row['lon']}&by=FOOT"

                matched_pois.append({
                    "poi_id": row['poi_id'],
                    "name": row['name'],
                    "category": row['category'],
                    "location": {
                        "lat": row['lat'],
                        "lon": row['lon']
                    },
                    "distance_m": round(dist_m, 1),
                    "score": round(road_affinity_score, 3),
                    "deep_links": {
                        "naver_map": naver_map_link,
                        "kakao_map": kakao_map_link
                    }
                })

        # 점수 기준 정렬
        matched_pois.sort(key=lambda x: x['score'], reverse=True)

        # 물리적 거리 기반 NMS 필터 (카테고리 불문 최소 150m 이상 이격 분산)
        deduped_pois = []
        for p in matched_pois:
            is_dup = False
            for accepted in deduped_pois:
                dy = (p["location"]["lat"] - accepted["location"]["lat"]) * 111000.0
                dx = (p["location"]["lon"] - accepted["location"]["lon"]) * 88800.0
                sep_dist = (dx**2 + dy**2)**0.5
                if sep_dist < 150.0:
                    is_dup = True
                    break
            if not is_dup:
                deduped_pois.append(p)
            if len(deduped_pois) >= top_k:
                break

        for rank, p in enumerate(deduped_pois, start=1):
            p['rank'] = rank
            p['recommendation_reason'] = self._generate_recommendation_reason(
                category=p['category'],
                dist_m=p.get('distance_m', 0.0)
            )

        return deduped_pois