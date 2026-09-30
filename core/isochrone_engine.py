"""
core/isochrone_engine.py
- GoldenStep Core Engine: 보행 도로망 기반 도달 권역(Isochrone Polygon) 산출 모듈
- 치매 노인 기준 속도: 2.40 km/h 및 지수 피로 감쇠 공식 적용 (Loss of Kinetics 모사)
- 기획서 v1.1(PB-02, PB-04) 5개 시간대 슬라이더 및 수색 면적 축소율(ARR) 산출 지원
"""

import math
import heapq
import logging
from typing import Dict, Any, List, Tuple
import networkx as nx
import osmnx as ox
from shapely.geometry import MultiPoint, Polygon, mapping
from core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GoldenStep-Isochrone")


class IsochroneEngine:
    def __init__(
        self,
        graph: nx.MultiDiGraph
    ):
        """
        :param graph: core.graph_loader에서 로드된 서울시 보행 도로망 MultiDiGraph
        """
        self.graph = graph

    def _calculate_decayed_distance(self, elapsed_hours: float, base_speed_kmh: float, fatigue_lambda: float) -> float:
        """
        지수 피로 감쇠 수식을 통한 실제 한계 도달 거리(m) 산출
        D(t) = (v0 / lambda) * (1 - exp(-lambda * t))
        """
        t = max(0.05, float(elapsed_hours))
        # km 단위 한계 거리
        d_km = (base_speed_kmh / fatigue_lambda) * (1.0 - math.exp(-fatigue_lambda * t))
        return d_km * 1000.0  # meter 변환

    def calculate_isochrone(
        self,
        center_lat: float,
        center_lon: float,
        elapsed_hours: float,
        base_speed_kmh: float = None,
        fatigue_lambda: float = None,
        turn_penalty_factor: float = 0.0
    ) -> Dict[str, Any]:
        """
        단일 경과 시간에 대한 도달 권역(Feature GeoJSON)을 산출합니다.
        """
        base_speed_kmh = base_speed_kmh or settings.BASE_WALK_VELOCITY_KMH
        fatigue_lambda = fatigue_lambda or settings.FATIGUE_DECAY_LAMBDA

        # 1. 지수 감쇠 한계 도달 거리(m) 계산
        max_distance_m = self._calculate_decayed_distance(elapsed_hours, base_speed_kmh, fatigue_lambda)

        # 2. 중심 좌표에서 가장 가까운 도로망 출발 노드 탐색
        start_node = ox.distance.nearest_nodes(self.graph, X=center_lon, Y=center_lat)

        # 3. 다익스트라(Dijkstra) 기반 도달 노드 탐색 (3D 경사도 및 도로 폭 가중치 반영)
        def _effective_distance_weight(u, v, edge_dict):
            # nx.MultiDiGraph에서는 edge_dict가 {0: data, 1: data...} 형태이므로 첫 번째 간선 정보 추출
            data = next(iter(edge_dict.values())) if isinstance(edge_dict, dict) and edge_dict and 0 in edge_dict else edge_dict
            
            length = float(data.get("length", 10.0))
            norm_width = float(data.get("norm_width", 1.0))
            slope = float(data.get("grade", 0.0))
            # 평지(0도)일 때 페널티 1.0, 경사가 가파를수록 페널티 증가 (속도 감소 모델 역산)
            slope_penalty = math.exp(3.50 * abs(slope + 0.05)) / math.exp(3.50 * 0.05)
            # 최종 유효 거리 = 물리적 거리 / 도로 폭 가중치 * 경사도 페널티
            return length * (1.0 / max(0.1, norm_width)) * slope_penalty

        if turn_penalty_factor <= 0.0:
            reachable_nodes_lengths = nx.single_source_dijkstra_path_length(
                self.graph,
                source=start_node,
                cutoff=max_distance_m,
                weight=_effective_distance_weight
            )
        else:
            reachable_nodes_lengths = self._dijkstra_with_turn_penalty(
                start_node, max_distance_m, turn_penalty_factor
            )

        # 4. 도달 노드 좌표 집합 수집 (lon, lat)
        node_points = [
            (self.graph.nodes[n]["x"], self.graph.nodes[n]["y"])
            for n in reachable_nodes_lengths.keys()
        ]

        # 5. 폴리곤 외곽선 생성
        polygon = self._generate_polygon(node_points, center_lat, center_lon, max_distance_m)

        # 6. 기하학적 수색 면적 축소율 (ARR, Area Reduction Rate) 계산
        # 단순 동심원 면적 (등속 직선 반경) 대비 도로망 실제 도달 면적 비교
        nominal_r_deg = (base_speed_kmh * 1000.0 * elapsed_hours) / 111000.0
        circle_area = math.pi * (nominal_r_deg ** 2)
        poly_area = polygon.area
        arr_val = max(0.0, min(98.0, (1.0 - (poly_area / max(circle_area, 1e-7))) * 100.0))

        # 표준 GeoJSON Feature 형태로 반환
        return {
            "type": "Feature",
            "geometry": mapping(polygon),
            "properties": {
                "elapsed_hours": elapsed_hours,
                "cutoff_distance_m": round(max_distance_m, 1),
                "reached_node_count": len(reachable_nodes_lengths),
                "area_reduction_rate": f"{arr_val:.1f}%",
                "arr_raw": round(arr_val / 100.0, 4)
            }
        }

    def calculate_timeline_isochrones(
        self,
        center_lat: float,
        center_lon: float,
        base_speed_kmh: float = None,
        fatigue_lambda: float = None,
        turn_penalty_factor: float = 0.0
    ) -> Dict[str, Dict[str, Any]]:
        """
        [기획서 v1.1 PB-04 준수]
        시간 슬라이더 5단계 규격: 지금(0.1h), 30분(0.5h), 1시간(1.0h), 3시간(3.0h), 6시간(6.0h)
        """
        base_speed_kmh = base_speed_kmh or settings.BASE_WALK_VELOCITY_KMH
        fatigue_lambda = fatigue_lambda or settings.FATIGUE_DECAY_LAMBDA

        steps = {
            "CURRENT": 0.1,    # 초기 6분
            "30M": 0.5,
            "1H": 1.0,
            "3H": 3.0,
            "6H": 6.0
        }

        results = {}
        for key, hours in steps.items():
            results[key] = self.calculate_isochrone(
                center_lat, center_lon, hours, base_speed_kmh, fatigue_lambda, turn_penalty_factor
            )
        return results

    def _generate_polygon(
        self,
        node_points: List[Tuple[float, float]],
        center_lat: float,
        center_lon: float,
        fallback_radius_m: float
    ) -> Polygon:
        """
        노드 좌표들로 도로망 완충 영역 다각형(Polygon)을 구성
        """
        if len(node_points) >= 3:
            mp = MultiPoint(node_points)
            poly = mp.convex_hull
            # 도로 링크 버퍼 (약 40m ≈ 0.00035도)
            poly = poly.buffer(0.00035).simplify(0.0001)
            if isinstance(poly, Polygon) and not poly.is_empty:
                return poly

        logger.warning("도달 노드가 부족하여 기본 완충 구역(Fallback)으로 대체합니다.")
        deg_radius = fallback_radius_m / 111000.0
        return MultiPoint([(center_lon, center_lat)]).buffer(deg_radius)


    def _dijkstra_with_turn_penalty(self, start_node, cutoff, k) -> Dict[Any, float]:
        queue = [(0.0, start_node, None)]
        node_costs = {}
        
        while queue:
            cost, u, p = heapq.heappop(queue)
            
            if u in node_costs and node_costs[u] <= cost:
                continue
                
            node_costs[u] = cost
            
            for v in self.graph.successors(u):
                edge_data = self.graph.get_edge_data(u, v)[0]
                length = edge_data.get("length", 1.0)
                
                penalty = 1.0
                if p is not None:
                    px, py = self.graph.nodes[p]['x'], self.graph.nodes[p]['y']
                    ux, uy = self.graph.nodes[u]['x'], self.graph.nodes[u]['y']
                    vx, vy = self.graph.nodes[v]['x'], self.graph.nodes[v]['y']
                    
                    v1_x, v1_y = ux - px, uy - py
                    v2_x, v2_y = vx - ux, vy - uy
                    
                    dot = v1_x*v2_x + v1_y*v2_y
                    mag1 = math.hypot(v1_x, v1_y)
                    mag2 = math.hypot(v2_x, v2_y)
                    
                    if mag1 > 0 and mag2 > 0:
                        cos_theta = dot / (mag1 * mag2)
                        cos_theta = max(-1.0, min(1.0, cos_theta))
                        
                        # 막힌 길 핑퐁 예외 처리
                        if self.graph.out_degree(u) <= 1:
                            penalty = 1.0
                        else:
                            penalty = 1.0 + k * (1.0 - cos_theta)
                            
                next_cost = cost + (length * penalty)
                if next_cost <= cutoff:
                    if v not in node_costs or next_cost < node_costs.get(v, float('inf')):
                        heapq.heappush(queue, (next_cost, v, u))
                        
        return node_costs

if __name__ == "__main__":
    import json
    from core.graph_loader import WalkGraphManager

    print(">> 도로망 캐시 로드 중...")
    loader = WalkGraphManager()
    G = loader.load_graph()

    engine = IsochroneEngine(graph=G)
    test_lat, test_lon = 37.5398, 126.9479  # 마포구 도화동
    res = engine.calculate_isochrone(test_lat, test_lon, 1.0, 2.40, 0.15)
    print(">> 1시간 도달 권역 연산 결과:")
    print(f"   - 수색 면적 축소율(ARR): {res['properties']['area_reduction_rate']}")
    print(f"   - 한계 거리: {res['properties']['cutoff_distance_m']} m")