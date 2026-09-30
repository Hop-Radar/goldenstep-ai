"""
core/graph_loader.py
- GoldenStep Core Engine: 서울특별시 전역(25개 자치구) 보행 도로망 추출, 정제, 캐싱 모듈
- 기준: Seoul, South Korea (약 605 km²)
- 노드/엣지 속성 정규화 및 계단(highway=steps) 가중치 0.05 부여
"""

import os
import time
import logging
import networkx as nx
import osmnx as ox
import sys
from pathlib import Path

# 설정 로드
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GoldenStep-GraphLoader")

# OSMnx 설정
ox.settings.log_console = False
ox.settings.use_cache = True

# 실제 서울 OSM 추출 태그(23종) 기반 최적화 가중치 테이블
HIGHWAY_WIDTH_WEIGHTS = {
    # 1. 보행량이 많고 폭이 넓은 간선도로 (치매 환자 강력 유인)
    "primary": 1.0,
    "primary_link": 0.9,
    "secondary": 0.8,
    "secondary_link": 0.7,
    "pedestrian": 0.9,      # 보행전용도로
    "footway": 0.7,         # 일반 보도/인도
    "crossing": 0.8,        # 횡단보도

    # 2. 중간 규모 연결 도로 및 대중교통 연계 보행로
    "tertiary": 0.6,
    "tertiary_link": 0.5,
    "busway": 0.5,          # 버스전용차로 인접 보도/정류장
    "unclassified": 0.5,    # 일반 생활 연결 도로
    "elevator": 0.6,        # 육교/지하도 엘리베이터

    # 3. 주택가 이면도로 및 일반 생활로
    "residential": 0.4,
    "living_street": 0.4,

    # 4. 좁은 골목길 및 비포장 산책길
    "path": 0.3,
    "service": 0.2,         # 아파트 단지/주차장 진입로
    "track": 0.2,           # 산로/임도
    "corridor": 0.3,        # 지하상가/건물 내 통로
    "road": 0.3,            # 미분류 도로

    # 5. 보행 환경이 극도로 열악한 구간 또는 진입 금지 도로
    "steps": 0.05,          # 계단
    "bridleway": 0.05,      # 승마/특수 도로
    "trunk_link": 0.01,     # 자동차전용도로 진출입로 (차단 목적)
    "trunk": 0.01           # 자동차전용도로 (차단 목적)
}


class WalkGraphManager:
    def __init__(
        self,
        cache_path: str = None,
        place_query: str = "Seoul, South Korea"
    ):
        """
        :param cache_path: 캐시 파일 저장 경로
        :param place_query: OSM 지정구역 검색어 (서울 전역)
        """
        self.cache_path = cache_path or settings.CACHE_GRAPH_PATH
        self.place_query = place_query

    def build_and_cache_subgraph(self) -> nx.MultiDiGraph:
        """서울특별시 전역의 보행 도로망을 다운로드하고 가중치 정규화 후 캐싱"""
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        logger.info(f"'{self.place_query}' 전역 보행 도로망 다운로드 시작 (대용량 작업)...")
        start_time = time.time()

        # 서울특별시 전역 보행 도로망 추출 (자동차 전용도로 배제)
        G = ox.graph_from_place(
            self.place_query,
            network_type="walk",
            simplify=True
        )
        logger.info(
            f"서울 전역 다운로드 완료 ({time.time() - start_time:.2f}초 소요) - 노드: {len(G.nodes):,}개, 엣지: {len(G.edges):,}개"
        )

        # 도로 설계 정규화 및 계단 가중치 주입
        self._preprocess_edges(G)

        # 최대 연결 컴포넌트 추출 (단절 파편 제거)
        G_cleaned = self._clean_isolated_components(G)

        # 로컬 캐시 파일 저장 (.graphml)
        logger.info(f"로컬 캐시 파일 저장 중: {self.cache_path}")
        ox.save_graphml(G_cleaned, filepath=self.cache_path)
        logger.info("서울 전역 캐시 파일 저장 완료.")

        return G_cleaned

    def load_graph(self) -> nx.MultiDiGraph:
        """캐시 파일이 있으면 즉시 로드하고, 없으면 신규 생성"""
        if not os.path.exists(self.cache_path):
            logger.warning(f"캐시 파일({self.cache_path})이 존재하지 않아 서울 전역 신규 다운로드를 시작합니다.")
            return self.build_and_cache_subgraph()

        start_time = time.time()
        G = ox.load_graphml(filepath=self.cache_path)
        logger.info(
            f"그래프 메모리 로드 완료 ({time.time() - start_time:.2f}초 소요 - 노드: {len(G.nodes):,}개, 엣지: {len(G.edges):,}개)"
        )
        return G

    def _preprocess_edges(self, G: nx.MultiDiGraph) -> None:
        """도로 경계망 정규화 가중치(norm_width) 주입 및 결측 길이 보정"""
        for u, v, k, data in G.edges(keys=True, data=True):
            hw_type = data.get("highway", "residential")
            if isinstance(hw_type, list):
                hw_type = hw_type[0]

            data["norm_width"] = float(HIGHWAY_WIDTH_WEIGHTS.get(hw_type, 0.4))

            if "length" not in data or not data["length"]:
                data["length"] = 10.0
            else:
                data["length"] = float(data["length"])

    def _clean_isolated_components(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """단절된 고립 노드 제거"""
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        return G.subgraph(largest_cc).copy()


if __name__ == "__main__":
    manager = WalkGraphManager()
    print(">> 서울 전역 보행 도로망 빌드/로드 시작...")
    graph = manager.load_graph()
    print(f">> 서울 전역 검증 성공: (노드 = {len(graph.nodes):,}개, 엣지 = {len(graph.edges):,}개)")
