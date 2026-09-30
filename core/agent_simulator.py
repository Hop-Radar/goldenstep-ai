"""
core/agent_simulator.py
- GoldenStep V2.0 Core: 확률론적 몬테카를로 에이전트 배회 시뮬레이터
- 에이전트 이질성(속도, 초기 방위각, 체류 임계점) 적용
- 누적 방향 전환 페널티 및 터널 시야(Tunnel Vision) 모사
- [수정] 거점 중심 반경 300m 완충 구역 발걸음 분포 및 유입 경로(high_probability_edges) 추출 기능 탑재
"""

import math
import random
import logging
from typing import Dict, Any, List, Tuple
import networkx as nx
import osmnx as ox
import sys
from pathlib import Path
from shapely.geometry import MultiPoint, Polygon, LineString, mapping

# 모듈 경로 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GoldenStep-AgentSimulator")


class Agent:
    def __init__(self, start_node: int, base_speed_kmh: float):
        self.current_node = start_node
        self.elapsed_hours = 0.0
        
        # 1. 신체 능력 편차 (Physical Condition Variance)
        self.speed_multiplier = random.uniform(0.85, 1.15)
        self.speed_kmh = base_speed_kmh * self.speed_multiplier
        
        # 2. 초기 지향성 및 터널 시야 (Initial Heading)
        self.target_heading = random.uniform(0.0, 360.0)
        
        # 3. 체류 임계점 편차 (Dwell Tolerance)
        self.dwell_tolerance = random.uniform(0.5, 1.5)
        
        # 상태 기록
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
        """두 점 사이의 절대 방위각(0~360) 계산"""
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
        """두 각도 간의 최소 차이 (0~180)"""
        diff = abs(angle1 - angle2) % 360
        return 360 - diff if diff > 180 else diff

    def _calculate_dynamic_speed(self, agent: Agent, slope: float) -> float:
        C_dem = 2.8590
        base_v = max(0.5, C_dem * math.exp(-3.50 * abs(slope + 0.05)))
        fatigue_lambda = settings.FATIGUE_DECAY_LAMBDA
        current_v = (base_v * agent.speed_multiplier) * math.exp(-fatigue_lambda * agent.elapsed_hours)
        return max(0.1, current_v)

    def simulate(
        self, 
        center_lat: float, 
        center_lon: float, 
        num_agents: int = 2000, 
        max_hours: float = 3.0,
        base_speed: float = 2.40,
        turn_penalty_gamma: float = 0.2
    ) -> Dict[str, Any]:
        """몬테카를로 에이전트 주행 및 궤적 수집"""
        start_node = ox.distance.nearest_nodes(self.graph, X=center_lon, Y=center_lat)
        agents = [Agent(start_node, base_speed) for _ in range(num_agents)]
        
        stopped_points = []
        visited_nodes = set([start_node])
        edge_visit_counts: Dict[Tuple[int, int], int] = {}
        
        for agent in agents:
            while agent.elapsed_hours < max_hours and not agent.is_stopped:
                u = agent.current_node
                successors = list(self.graph.successors(u))
                
                # 1. 막힌 길 처리
                if not successors:
                    agent.is_stopped = True
                    break
                
                # 2. U턴 지양
                valid_successors = [v for v in successors if v != agent.previous_node]
                if not valid_successors:
                    valid_successors = successors
                
                # 3. 전이 확률 계산
                weights = []
                u_coord = (self.graph.nodes[u]['x'], self.graph.nodes[u]['y'])
                
                for v in valid_successors:
                    v_coord = (self.graph.nodes[v]['x'], self.graph.nodes[v]['y'])
                    edge_data = self.graph.get_edge_data(u, v)[0]
                    length_m = edge_data.get("length", 10.0)
                    
                    edge_bearing = self._calculate_bearing(u_coord, v_coord)
                    angle_diff = self._angle_difference(agent.target_heading, edge_bearing)
                    
                    turn_ratio = angle_diff / 180.0
                    turn_penalty = (1.0 + 2.0 * (turn_ratio ** 2)) * (1.0 + turn_penalty_gamma * agent.accumulated_turns)
                    
                    cost = length_m * turn_penalty
                    weights.append(1.0 / max(cost, 0.1))
                
                # 4. 룰렛 휠 선택
                total_weight = sum(weights)
                probs = [w / total_weight for w in weights]
                chosen_v = random.choices(valid_successors, weights=probs, k=1)[0]
                
                # 엣지 카운트 누적
                edge_key = (min(u, chosen_v), max(u, chosen_v))
                edge_visit_counts[edge_key] = edge_visit_counts.get(edge_key, 0) + 1

                # 5. 상태 업데이트
                chosen_edge = self.graph.get_edge_data(u, chosen_v)[0]
                length_m = chosen_edge.get("length", 10.0)
                grade = chosen_edge.get("grade", 0.0)
                
                current_speed = self._calculate_dynamic_speed(agent, slope=grade)
                time_taken = (length_m / 1000.0) / current_speed
                agent.elapsed_hours += time_taken
                
                chosen_bearing = self._calculate_bearing(u_coord, (self.graph.nodes[chosen_v]['x'], self.graph.nodes[chosen_v]['y']))
                if self._angle_difference(agent.target_heading, chosen_bearing) > 45.0:
                    agent.accumulated_turns += 1
                    agent.update_heading(chosen_bearing)
                
                agent.previous_node = u
                agent.current_node = chosen_v
                agent.path.append(chosen_v)
                visited_nodes.add(chosen_v)
                
                # 6. 체류/정지 검사
                stop_prob = 1.0 - math.exp(-0.5 * agent.elapsed_hours / agent.dwell_tolerance)
                if agent.accumulated_turns >= 14:
                    stop_prob += 0.5
                    
                if random.random() < stop_prob:
                    agent.is_stopped = True
                    
            stopped_points.append((self.graph.nodes[agent.current_node]['x'], self.graph.nodes[agent.current_node]['y']))

        # Convex Hull 외곽 경계 생성
        base_geom = MultiPoint(stopped_points).convex_hull
        poly = base_geom.buffer(0.0005).simplify(0.0001)
        if not isinstance(poly, Polygon) and not poly.is_empty:
            poly = poly.buffer(0.0005)

        return {
            "type": "Feature",
            "geometry": mapping(poly) if not poly.is_empty else None,
            "properties": {
                "elapsed_hours": max_hours,
                "reached_node_count": len(visited_nodes),
                "agent_count": num_agents,
                "stopped_agents": len([a for a in agents if a.is_stopped])
            },
            "agents": agents,
            "edge_visit_counts": edge_visit_counts
        }

    def extract_poi_corridors(
        self,
        agents: List[Agent],
        top_pois: List[Dict[str, Any]],
        radius_m: float = 300.0
    ) -> Dict[str, Any]:
        """
        우선순위 거점(TOP 3) 중심 반경 300m 이내 발걸음 분포 및
        해당 거점으로 유입된 이동 경로를 이용 빈도순으로 추출하여 GeoJSON LineString 생성
        """
        if not top_pois or not agents:
            return {"type": "FeatureCollection", "features": []}

        corridor_edge_counts: Dict[Tuple[int, int], int] = {}
        poi_zones = []

        # 1. 각 거점별 반경 300m 내 노드 집합 사전 탐색 (유클리드 근사)
        for poi in top_pois:
            p_lat = poi["location"]["lat"]
            p_lon = poi["location"]["lon"]
            
            # 위도 1도 ≈ 111,000m, 경도 1도 ≈ 88,800m
            lat_delta = radius_m / 111000.0
            lon_delta = radius_m / 88800.0
            
            nearby_nodes = set()
            for n, data in self.graph.nodes(data=True):
                if (p_lat - lat_delta <= data['y'] <= p_lat + lat_delta and
                    p_lon - lon_delta <= data['x'] <= p_lon + lon_delta):
                    dy = (data['y'] - p_lat) * 111000.0
                    dx = (data['x'] - p_lon) * 88800.0
                    if (dx**2 + dy**2)**0.5 <= radius_m:
                        nearby_nodes.add(n)
            
            poi_zones.append({
                "rank": poi.get("rank", 1),
                "name": poi.get("name", ""),
                "nearby_nodes": nearby_nodes
            })

        # 2. 거점 300m에 도달한 에이전트들의 경로 엣지 집계
        for agent in agents:
            agent_nodes = set(agent.path)
            reached_pois = [pz for pz in poi_zones if bool(agent_nodes & pz["nearby_nodes"])]
            
            # 최소 하나의 거점에 유입된 에이전트의 경로만 추출
            if reached_pois:
                for i in range(len(agent.path) - 1):
                    u = agent.path[i]
                    v = agent.path[i + 1]
                    edge_key = (min(u, v), max(u, v))
                    corridor_edge_counts[edge_key] = corridor_edge_counts.get(edge_key, 0) + 1

        # 거점에 도달한 에이전트가 극히 적을 경우 전체 방문 엣지 중 상위 엣지로 폴백
        if not corridor_edge_counts:
            for agent in agents:
                for i in range(len(agent.path) - 1):
                    u = agent.path[i]
                    v = agent.path[i + 1]
                    edge_key = (min(u, v), max(u, v))
                    corridor_edge_counts[edge_key] = corridor_edge_counts.get(edge_key, 0) + 1

        # 3. 빈도순 정렬 및 상위 도로망 추출 (최대 50개 링크)
        sorted_edges = sorted(corridor_edge_counts.items(), key=lambda x: x[1], reverse=True)
        top_edges = sorted_edges[:50]
        max_visits = top_edges[0][1] if top_edges else 1

        edge_features = []
        for (eu, ev), count in top_edges:
            edge_data = self.graph.get_edge_data(eu, ev, 0) or self.graph.get_edge_data(ev, eu, 0)
            if not edge_data:
                continue

            if "geometry" in edge_data:
                geom = edge_data["geometry"]
            else:
                u_node = self.graph.nodes[eu]
                v_node = self.graph.nodes[ev]
                geom = LineString([(u_node["x"], u_node["y"]), (v_node["x"], v_node["y"])])

            prob = round(count / max_visits, 3)
            # 프론트엔드 스타일 매핑 규격 준수 (CRITICAL, HIGH, MODERATE)
            priority = "CRITICAL" if prob >= 0.7 else ("HIGH" if prob >= 0.4 else "MODERATE")

            edge_features.append({
                "type": "Feature",
                "properties": {
                    "u": eu,
                    "v": ev,
                    "visit_count": count,
                    "probability": prob,
                    "priority": priority
                },
                "geometry": mapping(geom)
            })

        return {
            "type": "FeatureCollection",
            "features": edge_features
        }