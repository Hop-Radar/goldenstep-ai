"""
core/isochrone_engine.py
- GoldenStep Core Engine: 보행 도로망 기반 A1(물리적 최대 도달 영역) 및 A2(보행 가능 네트워크 영역) 산출 모듈
- Kinematic Baseline: 등속 기반 최대 도달 반경 R_max = v0 * t 적용 (임의의 피로 감쇠 lambda 배제)
- A2 영역: 중심선 기준 편측 5m (총 도로 폭 10m) 버퍼 합집합(Unary Union) 면적 연산
"""

import math
import heapq
import logging
from typing import Dict, Any, List, Tuple
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, MultiPoint, Polygon, LineString, mapping
from shapely.ops import unary_union
from core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GoldenStep-Isochrone")


class IsochroneEngine:
    def __init__(self, graph: nx.MultiDiGraph):
        """
        :param graph: core.graph_loader에서 로드된 서울시 보행 도로망 MultiDiGraph
        """
        self.graph = graph

    def calculate_a1_max_distance(self, elapsed_hours: float, base_speed_kmh: float = None) -> float:
        """
        [A1 Kinematic Baseline] 등속 보행 기준 물리적 최대 도달 한계 거리(m) 산출
        R_max = v0 * t
        """
        t = max(0.01, float(elapsed_hours))
        v0 = base_speed_kmh or settings.BASE_WALK_VELOCITY_KMH
        # km -> meter 변환
        return v0 * t * 1000.0

    def calculate_a2_network(
        self,
        center_lat: float,
        center_lon: float,
        r_max_m: float
    ) -> Dict[str, Any]:
        """
        [A2 Walkable Network Area]
        A1 원형 반경 내의 보행 가능 도로망 간선에 편측 5m (총 폭 10m) 버퍼를 적용한 통합 영역 산출
        """
        # 위도 1도 ≈ 111,000m, 경도 1도 ≈ 88,800m
        lat_delta = r_max_m / 111000.0
        lon_delta = r_max_m / 88800.0

        # WGS84 좌표계 기준의 타원체 거리 보정을 고려한 중심 원형 버퍼 생성 (바운딩용)
        center_pt = Point(center_lon, center_lat)
        deg_radius = r_max_m / 111000.0
        a1_circle_approx = center_pt.buffer(deg_radius)

        buffered_edge_geoms = []
        total_length_m = 0.0
        edge_count = 0

        # 도로망 간선 필터링 및 10m 버퍼링 (위경도 환산: 5m ≈ 0.000045도)
        # 보다 정밀한 공간 연산을 위해 국소 메트릭 버퍼 환산값 적용
        buffer_deg = settings.STREET_BUFFER_HALF_WIDTH_M / 111000.0

        for u, v, k, data in self.graph.edges(keys=True, data=True):
            u_node = self.graph.nodes[u]
            v_node = self.graph.nodes[v]
            
            # 중심점 기준 1차 Bounding Box 검사
            if not (
                min(u_node['y'], v_node['y']) >= center_lat - lat_delta and
                max(u_node['y'], v_node['y']) <= center_lat + lat_delta and
                min(u_node['x'], v_node['x']) >= center_lon - lon_delta and
                max(u_node['x'], v_node['x']) <= center_lon + lon_delta
            ):
                continue

            # 간선 기하 추출
            if "geometry" in data:
                edge_geom = data["geometry"]
            else:
                edge_geom = LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])

            # A1 원형 영역과의 교차 검사
            if a1_circle_approx.intersects(edge_geom):
                buffered_edge_geoms.append(edge_geom.buffer(buffer_deg))
                total_length_m += float(data.get("length", 10.0))
                edge_count += 1

        if not buffered_edge_geoms:
            logger.warning("A1 반경 내에 추출된 도로 엣지가 없습니다. 기본 반경 폴리곤으로 대체합니다.")
            a2_polygon = a1_circle_approx
        else:
            # 겹치는 버퍼 다각형 합집합(Unary Union) 병합
            a2_polygon = unary_union(buffered_edge_geoms)

        # 면적 계산 (km^2 단위 변환)
        # 위경도 도 단위 면적을 m^2로 환산: deg_area * (111000 * 88800)
        area_km2 = (a2_polygon.area * 111000.0 * 88800.0) / 1_000_000.0
        a1_area_km2 = (math.pi * (r_max_m ** 2)) / 1_000_000.0

        arr1_val = max(0.0, min(99.0, (1.0 - (area_km2 / max(a1_area_km2, 1e-7))) * 100.0))

        return {
            "geometry": a2_polygon,
            "area_km2": round(area_km2, 4),
            "a1_area_km2": round(a1_area_km2, 4),
            "total_length_km": round(total_length_m / 1000.0, 2),
            "edge_count": edge_count,
            "arr1_percent": round(arr1_val, 2)
        }

    def calculate_isochrone(
        self,
        center_lat: float,
        center_lon: float,
        elapsed_hours: float,
        base_speed_kmh: float = None,
        turn_penalty_factor: float = 0.0
    ) -> Dict[str, Any]:
        """
        단일 경과 시간에 대한 A1(최대한계) 및 A2(보행로) 기반 도달 권역 Feature GeoJSON 산출
        """
        base_speed_kmh = base_speed_kmh or settings.BASE_WALK_VELOCITY_KMH

        # 1. 등속 기준 한계 도달 거리(m) 계산 (A1 경계)
        r_max_m = self.calculate_a1_max_distance(elapsed_hours, base_speed_kmh)

        # 2. 중심 좌표에서 가장 가까운 도로망 출발 노드 탐색
        start_node = ox.distance.nearest_nodes(self.graph, X=center_lon, Y=center_lat)

        # 3. 다익스트라(Dijkstra) 기반 도달 노드 탐색
        def _effective_distance_weight(u, v, edge_dict):
            data = next(iter(edge_dict.values())) if isinstance(edge_dict, dict) and edge_dict and 0 in edge_dict else edge_dict
            
            length = float(data.get("length", 10.0))
            norm_width = float(data.get("norm_width", 1.0))
            slope = float(data.get("grade", 0.0))
            # 평지 기준 상대적 Tobler 페널티
            slope_penalty = math.exp(3.50 * abs(slope + 0.05)) / math.exp(3.50 * 0.05)
            return length * (1.0 / max(0.1, norm_width)) * slope_penalty

        if turn_penalty_factor <= 0.0:
            reachable_nodes_lengths = nx.single_source_dijkstra_path_length(
                self.graph,
                source=start_node,
                cutoff=r_max_m,
                weight=_effective_distance_weight
            )
        else:
            reachable_nodes_lengths = self._dijkstra_with_turn_penalty(
                start_node, r_max_m, turn_penalty_factor
            )

        # 4. 도달 노드 좌표 집합 수집
        node_points = [
            (self.graph.nodes[n]["x"], self.graph.nodes[n]["y"])
            for n in reachable_nodes_lengths.keys()
        ]

        # 5. 도로망 완충 영역 다각형 생성
        polygon = self._generate_polygon(node_points, center_lat, center_lon, r_max_m)

        # 6. A2 네트워크 연산 및 1차 면적 축소율 도출
        a2_info = self.calculate_a2_network(center_lat, center_lon, r_max_m)

        return {
            "type": "Feature",
            "geometry": mapping(polygon),
            "properties": {
                "elapsed_hours": elapsed_hours,
                "cutoff_distance_m": round(r_max_m, 1),
                "reached_node_count": len(reachable_nodes_lengths),
                "a1_area_km2": a2_info["a1_area_km2"],
                "a2_area_km2": a2_info["area_km2"],
                "a2_total_length_km": a2_info["total_length_km"],
                "a2_edge_count": a2_info["edge_count"],
                "area_reduction_rate_a2": f"{a2_info['arr1_percent']:.1f}%"
            }
        }

    def _generate_polygon(
        self,
        node_points: List[Tuple[float, float]],
        center_lat: float,
        center_lon: float,
        fallback_radius_m: float
    ) -> Polygon:
        """노드 좌표들로 도로망 완충 영역 다각형을 구성"""
        if len(node_points) >= 3:
            mp = MultiPoint(node_points)
            poly = mp.convex_hull
            # 도로 링크 완충 버퍼
            poly = poly.buffer(0.00035).simplify(0.0001)
            if isinstance(poly, Polygon) and not poly.is_empty:
                return poly

        logger.warning("도달 노드가 부족하여 기본 완충 구역(Fallback)으로 대체합니다.")
        deg_radius = fallback_radius_m / 111000.0
        return Point(center_lon, center_lat).buffer(deg_radius)

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
                        
                        if self.graph.out_degree(u) <= 1:
                            penalty = 1.0
                        else:
                            penalty = 1.0 + k * (1.0 - cos_theta)
                            
                next_cost = cost + (length * penalty)
                if next_cost <= cutoff:
                    if v not in node_costs or next_cost < node_costs.get(v, float('inf')):
                        heapq.heappush(queue, (next_cost, v, u))
                        
        return node_costs