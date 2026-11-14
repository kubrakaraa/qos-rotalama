import random
from Node import Node
from Link import Link
from Demand import Demand
from collections import deque


class Graph:
    def __init__(self, node_csv, link_csv, demand_csv):
        self.nodes = {}  # {node_id: Node}
        self.adj = {}  # {node_id: [Link, Link, ...]}
        self.link_map = {}  # {key: Link}
        self.demand_map = {}  # {key: Demand}

        self.load_nodes(node_csv)
        self.load_links(link_csv)
        self.load_demands(demand_csv)

    # ============== CSV YUKLEYICILER ==============

    def load_nodes(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # İlk satırı atla (header)
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split(';')
            node_id = int(parts[0].strip())
            delay = float(parts[1].strip().replace(',', '.'))
            rel = float(parts[2].strip().replace(',', '.'))

            self.nodes[node_id] = Node(node_id, delay, rel)
            self.adj[node_id] = []

    def load_links(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split(';')
            s = int(parts[0].strip())
            d = int(parts[1].strip())
            bw = int(parts[2].strip())
            delay = float(parts[3].strip().replace(',', '.'))
            rel = float(parts[4].strip().replace(',', '.'))

            link = Link(s, d, delay, rel, bw)
            reverse = Link(d, s, delay, rel, bw)

            self.adj[s].append(link)
            self.adj[d].append(reverse)

            self.link_map[self.key(s, d)] = link
            self.link_map[self.key(d, s)] = reverse

    def load_demands(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split(';')
            s = int(parts[0].strip())
            d = int(parts[1].strip())
            bw = int(parts[2].strip())

            self.demand_map[self.key(s, d)] = Demand(s, d, bw)

    # ============== HELPER METODLAR ==============

    def key(self, a, b):
        return f"{a}_{b}"

    def get_link(self, s, d):
        return self.link_map.get(self.key(s, d))

    def get_demand(self, s, d):
        return self.demand_map.get(self.key(s, d))

    def set_demand(self, s, d, bw):
        self.demand_map[self.key(s, d)] = Demand(s, d, bw)

    def remove_demand(self, s, d):
        key = self.key(s, d)
        if key in self.demand_map:
            del self.demand_map[key]

    # ============== RASTGELE YOL BULMA ==============

    def generate_random_path(self, start, end):
        visited = set()
        path_stack = []

        found = self._visit_graph(start, end, visited, path_stack)

        if found:
            return list(path_stack)
        return []

    def _visit_graph(self, cur, end, visited, path_stack):
        path_stack.append(cur)
        visited.add(cur)

        if cur == end:
            return True

        # Komşuları bul
        neighbors = []
        for link in self.adj[cur]:
            if link.destination not in visited:
                neighbors.append(link.destination)

        # Rastgele sırala
        random.shuffle(neighbors)

        # DFS ile dene
        for nxt in neighbors:
            if self._visit_graph(nxt, end, visited, path_stack):
                return True

        # Backtrack
        visited.remove(cur)
        path_stack.pop()
        return False

    # ============== BW-VALID YOL BULMA ==============

    def find_bw_valid_path(self, start, end):
        demand = self.get_demand(start, end)
        req = demand.require_bandwidth if demand else 0

        # Demand yoksa normal path
        if req <= 0:
            return self.generate_random_path(start, end)

        visited = set()
        path = []

        found = self._dfs_bw(start, end, req, visited, path)

        if found:
            return list(path)
        return []


    def _dfs_bw(self, cur, end, req_bw, visited, path):
        visited.add(cur)
        path.append(cur)

        if cur == end:
            return True

        neighbors = self.adj.get(cur, [])

        # 1️⃣ Önce end var mı bak
        for link in neighbors:
            if link.destination == end and link.bandwidth >= req_bw:
                path.append(end)
                return True

        # 2️⃣ Sonra diğerlerini rastgele dene
        shuffled = list(neighbors)
        random.shuffle(shuffled)

        for link in shuffled:
            if link.bandwidth < req_bw:
                continue

            nxt = link.destination
            if nxt in visited or nxt == end:
                continue

            if self._dfs_bw(nxt, end, req_bw, visited, path):
                return True

        path.pop()
        visited.remove(cur)
        return False

    def find_bw_valid_path_bfs(self, start, end):
        demand = self.get_demand(start, end)
        req_bw = demand.require_bandwidth if demand else 0

        queue = deque()
        queue.append((start, [start]))
        visited = {start}

        while queue:
            cur, path = queue.popleft()

            if cur == end:
                return path

            for link in self.adj.get(cur, []):
                if link.bandwidth < req_bw:
                    continue

                nxt = link.destination
                if nxt in visited:
                    continue

                visited.add(nxt)
                queue.append((nxt, path + [nxt]))

        return []



'''
    def _dfs_bw(self, cur, end, req_bw, visited, path):
        visited.add(cur)
        path.append(cur)

        if cur == end:
            return True

        # Komşuları al ve karıştır
        neighbors = self.adj.get(cur, [])
        shuffled = list(neighbors)
        random.shuffle(shuffled)

        for link in shuffled:
            # BW kontrolü

            if link.bandwidth < req_bw:
                continue


            nxt = link.destination
            if nxt in visited:
                continue

            if self._dfs_bw(nxt, end, req_bw, visited, path):
                return True

        # Backtrack
        path.pop()
        visited.remove(cur)
        return False
'''





# ============== TEST ==============
if __name__ == "__main__":
    try:
        g = Graph("node.csv", "link.csv", "demand.csv")
        print("✅ Graf başarıyla yüklendi!")
        print(f"Node sayısı: {len(g.nodes)}")
        print(f"Link sayısı: {len(g.link_map) // 2}")
        print(f"Demand sayısı: {len(g.demand_map)}")

        # Test: Rastgele path
        path = g.generate_random_path(0, 5)
        print(f"\nRastgele path (0→5): {path}")

        # Test: BW-valid path
        bw_path = g.find_bw_valid_path(0, 5)
        print(f"BW-valid path (0→5): {bw_path}")

    except Exception as e:
        print(f"❌ Hata: {e}")