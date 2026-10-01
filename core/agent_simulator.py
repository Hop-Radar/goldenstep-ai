"""
core/agent_simulator.py
- GoldenStep V2.0 Core: 몬테카를로 에이전트 기반 보행 전이 및 이동 권역(A3) 산출 모듈
- 다항 로짓 기반 갈림길 의사결정 (직진성 S_ij, 도로폭 W_j, Tobler 상대비용 결합)
- [수정] 스텝별 조기 정지(Early Stop) 주사위 폐기 -> 목표 시간까지 충실 이동 (Go until stuck)
- [수정] 에이전트 최종 체류/도달 도로(terminal_edge_counts) 신규 집계 및 반환
"""

import math
import random
import logging
from typing import Dict, Any, List, Tuple
import networkx as nx
import osmnx as ox
from shapely.geometry import Point, MultiPoint, Polygon, LineString, mapping
from shapely.ops import unary_union
from core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GoldenStep-AgentSimulator")


class Agent:
    def __init__(self, start_node: int, base_speed_kmh: float):
        self.current_node = start_node
        self.elapsed_hours = 0.0
        
        # 신체 이동 편차 (개인별 미세 변동 ±15%)
        self.speed_multiplier = random.uniform(0.85, 1.15)
        self.speed_kmh = base_speed_kmh * self.speed_multiplier
        
        # 초기 진행 방위각 (0~360도 균등 분포)
        self.target_heading = random.uniform(0.0, 360.0)
        
        # 개인별 최대 보행 지구력 편차 (목표 시간 대비 85% ~ 115%)
        self.endurance_factor = random.uniform(0.85, 1.15)
        
        # 상태 제어 플래그
        self.is_stopped = False
        self.accumulated_turns = 0
        self.path = [start_node]
        self.previous_node = None

    def update_heading(self, new_heading: float):
        self.target_heading = new_heading


class MultiTypeRandomWalkSimulator:
    def __init__(self, graph: nx.MultiDiGraph):
        self.graph = graph

    def _calculate_bearing(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """두 점 사이의 절대 방위각(0~360도) 계산"""
        lon1, lat1 = p1
        lon2, lat2 = p2
        dLon = math.radians(lon2 - lon1)
        lat1 = math.radians(lat1)
        lat2 = math.radians(lat2)
        
        y = math.sin(dLon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
        brng = math.degrees(math.atan2(y, x))
        return (brng + 360) % 360

    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """두 방위각 간의 최소 회전각 (0~180도)"""
        diff = abs(angle1 - angle2) % 360
        return 360 - diff if diff > 180 else diff

    def _calculate_edge_speed(self, agent: Agent, slope: float) -> float:
        """Tobler 보행 감속이 반영된 실효 통과 속도(km/h) 산출"""
        slope_factor = math.exp(-3.50 * abs(slope + 0.05)) / math.exp(-3.50 * 0.05)
        current_v = agent.speed_kmh * slope_factor
        return max(0.2, current_v)

    def simulate(
        self, 
        center_lat: float, 
        center_lon: float, 
        num_agents: int = None, 
        max_hours: float = 2.0,
        base_speed: float = None,
        beta_dp: float = 1.0,
        beta_cost: float = 1.0,
        beta_wid: float = 1.0
    ) -> Dict[str, Any]:
        """
        몬테카를로 에이전트 주행 및 누적 통과 도로망(A3), 최종 체류 도로 산출
        """
        num_agents = num_agents or settings.NUM_SIMULATION_AGENTS
        base_speed = base_speed or settings.BASE_WALK_VELOCITY_KMH
        
        start_node = ox.distance.nearest_nodes(self.graph, X=center_lon, Y=center_lat)
        agents = [Agent(start_node, base_speed) for _ in range(num_agents)]
        
        stopped_points: List[Tuple[float, float]] = []
        visited_nodes = set([start_node])
        edge_visit_counts: Dict[Tuple[int, int], int] = {}
        terminal_edge_counts: Dict[Tuple[int, int], int] = {}
        
        for agent in agents:
            effective_limit = max_hours * agent.endurance_factor

            while agent.elapsed_hours < effective_limit and not agent.is_stopped:
                u = agent.current_node
                successors = list(self.graph.successors(u))
                
                # 1. 막다른 길 도달 시 (Koester: Stuck)
                if not successors:
                    agent.is_stopped = True
                    break
                
                valid_successors = successors
                
                # 2. 다항 로짓 효용 점수 및 소프트맥스 전이 확률 연산
                utilities = []
                u_coord = (self.graph.nodes[u]['x'], self.graph.nodes[u]['y'])
                
                for v in valid_successors:
                    v_coord = (self.graph.nodes[v]['x'], self.graph.nodes[v]['y'])
                    edge_dict = self.graph.get_edge_data(u, v)
                    edge_data = next(iter(edge_dict.values())) if isinstance(edge_dict, dict) and edge_dict else {}
                    
                    length_m = float(edge_data.get("length", 10.0))
                    grade = float(edge_data.get("grade", 0.0))
                    norm_width = float(edge_data.get("norm_width", 0.5))
                    
                    # 피처 1: 직진 코사인 유사도
                    edge_bearing = self._calculate_bearing(u_coord, v_coord)
                    angle_diff = self._angle_difference(agent.target_heading, edge_bearing)
                    s_ij = math.cos(math.radians(angle_diff))
                    
                    # 피처 2: Tobler 기반 상대 이동 비용
                    relative_cost = (length_m / 10.0) * math.exp(3.50 * abs(grade + 0.05))
                    
                    # 피처 3: 백트래킹 페널티
                    reversal_penalty = 2.0 if (v == agent.previous_node and len(successors) > 1) else 0.0
                    
                    u_ij = (beta_dp * s_ij) - (beta_cost * (relative_cost / 10.0)) + (beta_wid * norm_width) - reversal_penalty
                    utilities.append(u_ij)
                
                max_u = max(utilities)
                exp_u = [math.exp(u_val - max_u) for u_val in utilities]
                sum_exp = sum(exp_u)
                probs = [e / sum_exp for e in exp_u]
                
                chosen_v = random.choices(valid_successors, weights=probs, k=1)[0]
                
                # 통과 엣지 카운트
                edge_key = (min(u, chosen_v), max(u, chosen_v))
                edge_visit_counts[edge_key] = edge_visit_counts.get(edge_key, 0) + 1

                # 상태 갱신
                chosen_dict = self.graph.get_edge_data(u, chosen_v)
                chosen_edge = next(iter(chosen_dict.values())) if isinstance(chosen_dict, dict) and chosen_dict else {}
                length_m = float(chosen_edge.get("length", 10.0))
                grade = float(chosen_edge.get("grade", 0.0))
                
                step_speed = self._calculate_edge_speed(agent, slope=grade)
                time_spent = (length_m / 1000.0) / step_speed
                agent.elapsed_hours += time_spent
                
                chosen_bearing = self._calculate_bearing(u_coord, (self.graph.nodes[chosen_v]['x'], self.graph.nodes[chosen_v]['y']))
                if self._angle_difference(agent.target_heading, chosen_bearing) > 45.0:
                    agent.accumulated_turns += 1
                    agent.update_heading(chosen_bearing)
                
                agent.previous_node = u
                agent.current_node = chosen_v
                agent.path.append(chosen_v)
                visited_nodes.add(chosen_v)

            # 에이전트 최종 체류 도로(Terminal Edge) 집계
            if len(agent.path) >= 2:
                t_u, t_v = agent.path[-2], agent.path[-1]
                t_key = (min(t_u, t_v), max(t_u, t_v))
                terminal_edge_counts[t_key] = terminal_edge_counts.get(t_key, 0) + 1
            
            stopped_points.append((self.graph.nodes[agent.current_node]['x'], self.graph.nodes[agent.current_node]['y']))

        # A3 다각형 생성
        a3_polygon, a3_area_km2 = self._generate_a3_polygon(edge_visit_counts)

        return {
            "type": "Feature",
            "geometry": mapping(a3_polygon) if not a3_polygon.is_empty else None,
            "properties": {
                "elapsed_hours": max_hours,
                "reached_node_count": len(visited_nodes),
                "agent_count": num_agents,
                "stopped_agents": len(agents),
                "unique_edges_traversed": len(edge_visit_counts),
                "a3_area_km2": round(a3_area_km2, 4)
            },
            "agents": agents,
            "stopped_points": stopped_points,
            "edge_visit_counts": edge_visit_counts,
            "terminal_edge_counts": terminal_edge_counts,
            "a3_geometry": a3_polygon
        }

    def _generate_a3_polygon(self, edge_visit_counts: Dict[Tuple[int, int], int]) -> Tuple[Polygon, float]:
        """고유 간선 선형에 편측 5m 버퍼를 적용한 A3 합집합 다각형 및 면적 산출"""
        if not edge_visit_counts:
            return Polygon(), 0.0

        buffer_deg = settings.STREET_BUFFER_HALF_WIDTH_M / 111000.0
        edge_buffers = []

        for (u, v) in edge_visit_counts.keys():
            edge_dict = self.graph.get_edge_data(u, v) or self.graph.get_edge_data(v, u)
            if not edge_dict:
                continue
            data = next(iter(edge_dict.values())) if isinstance(edge_dict, dict) and edge_dict else {}
            
            if "geometry" in data:
                geom = data["geometry"]
            else:
                u_node = self.graph.nodes[u]
                v_node = self.graph.nodes[v]
                geom = LineString([(u_node['x'], u_node['y']), (v_node['x'], v_node['y'])])

            edge_buffers.append(geom.buffer(buffer_deg))

        a3_union = unary_union(edge_buffers)
        area_km2 = (a3_union.area * 111000.0 * 88800.0) / 1_000_000.0
        return a3_union, area_km2

    def extract_poi_corridors(
        self,
        agents: List[Agent],
        top_pois: List[Dict[str, Any]],
        radius_m: float = 200.0
    ) -> Dict[str, Any]:
        """
        [완결판] 출발지에서 우선순위 거점(TOP 1~3) 각각의 문앞까지 이어지는
        100% 끊김 없는 연속 보행 축선 및 거점 진입 도로망 추출
        """
        if not top_pois or not agents:
            return {"type": "FeatureCollection", "features": []}

        start_node = agents[0].path[0]
        selected_edges: Dict[Tuple[int, int], Dict[str, Any]] = {}

        # 1. 2,000명 에이전트 전체의 고유 통과 빈도 사전 맵 구축
        edge_popularity: Dict[Tuple[int, int], int] = {}
        for a in agents:
            for i in range(len(a.path) - 1):
                u, v = a.path[i], a.path[i + 1]
                ek = (min(u, v), max(u, v))
                edge_popularity[ek] = edge_popularity.get(ek, 0) + 1

        # 2. 각 거점(Rank 1, 2, 3)별로 출발지 -> 거점 문앞까지의 '연속 경로' 직접 연결
        for poi in top_pois:
            rank = poi.get("rank", 1)
            p_lat = poi["location"]["lat"]
            p_lon = poi["location"]["lon"]

            # 거점에서 가장 가까운 실제 도로망 진입 노드 탐색
            closest_node = None
            min_d = float('inf')
            lat_delta = 0.003
            lon_delta = 0.004

            nearby_nodes = set()
            for n, data in self.graph.nodes(data=True):
                if (p_lat - lat_delta <= data['y'] <= p_lat + lat_delta and
                    p_lon - lon_delta <= data['x'] <= p_lon + lon_delta):
                    dy = (data['y'] - p_lat) * 111000.0
                    dx = (data['x'] - p_lon) * 88800.0
                    d = (dx**2 + dy**2)**0.5
                    if d <= radius_m:
                        nearby_nodes.add(n)
                    if d < min_d:
                        min_d = d
                        closest_node = n

            # [핵심] 출발지 -> 거점 진입 노드까지의 끊김 없는 연속 실선 추출
            if closest_node and nx.has_path(self.graph, start_node, closest_node):
                # 에이전트 통과 빈도를 반영한 최적 보행 축선 경로 산출
                try:
                    # 빈도가 높은 도로일수록 이동 가중치(cost)를 낮춰 에이전트 다수 통과로를 선택
                    path = nx.shortest_path(
                        self.graph, 
                        source=start_node, 
                        target=closest_node,
                        weight=lambda u, v, d: float(d.get("length", 10.0)) / (1.0 + math.log1p(edge_popularity.get((min(u, v), max(u, v)), 0)))
                    )
                except Exception:
                    path = nx.shortest_path(self.graph, source=start_node, target=closest_node, weight="length")

                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    ek = (min(u, v), max(u, v))
                    pop = edge_popularity.get(ek, 50)
                    if ek not in selected_edges:
                        selected_edges[ek] = {"count": pop, "rank": rank}
                    else:
                        selected_edges[ek]["count"] = max(selected_edges[ek]["count"], pop)

            # [핵심] 거점 문앞 150m 주변 도로망 완충 연결
            for u, v, data in self.graph.edges(nearby_nodes, data=True):
                if u in nearby_nodes and v in nearby_nodes:
                    ek = (min(u, v), max(u, v))
                    pop = edge_popularity.get(ek, 20)
                    if ek not in selected_edges:
                        selected_edges[ek] = {"count": pop, "rank": rank}

        if not selected_edges:
            return {"type": "FeatureCollection", "features": []}

        max_c = max(d["count"] for d in selected_edges.values()) if selected_edges else 1

        edge_features = []
        for (eu, ev), meta in selected_edges.items():
            edge_dict = self.graph.get_edge_data(eu, ev) or self.graph.get_edge_data(ev, eu)
            if not edge_dict:
                continue
            data = next(iter(edge_dict.values())) if isinstance(edge_dict, dict) and edge_dict else {}

            if "geometry" in data:
                geom = data["geometry"]
            else:
                u_node = self.graph.nodes[eu]
                v_node = self.graph.nodes[ev]
                geom = LineString([(u_node["x"], u_node["y"]), (v_node["x"], v_node["y"])])

            prob = round(meta["count"] / max_c, 3)
            # 1순위 경로이거나 빈도가 높으면 CRITICAL
            priority = "CRITICAL" if meta["rank"] == 1 or prob >= 0.6 else ("HIGH" if prob >= 0.3 else "MODERATE")

            edge_features.append({
                "type": "Feature",
                "properties": {
                    "u": eu,
                    "v": ev,
                    "target_poi_rank": meta["rank"],
                    "visit_count": meta["count"],
                    "probability": prob,
                    "priority": priority
                },
                "geometry": mapping(geom)
            })

        return {
            "type": "FeatureCollection",
            "features": edge_features
        }