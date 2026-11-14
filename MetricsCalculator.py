import random
import math


class MetricsCalculator:
    PENALTY = 1e9

    # ✅ DFS'nin takılmaması için adım limiti
    DFS_STEP_LIMIT = 300  # 10k-50k arası deneyebilirsin

    # ============== 1) TRUE NODE'LAR ÜZERİNDE YOL BUL ==============

    @staticmethod
    def decode_to_path(chromosome, graph, start, end):
        if chromosome is None or chromosome.genes is None:
            return []

        n = len(chromosome.genes)
        if start < 0 or end < 0 or start >= n or end >= n:
            return []

        # start/end TRUE olmak zorunda
        if not chromosome.genes[start] or not chromosome.genes[end]:
            return []

        path = [start]
        visited = {start}

        # ✅ mutable counter (Python trick)
        steps = [0]

        found = MetricsCalculator._dfs_true_only_random(
            graph, chromosome.genes, start, end, visited, path, steps
        )

        if not found:
            return []
        return list(path)

    @staticmethod
    def _dfs_true_only_random(graph, genes, cur, end, visited, path, steps):
        # ✅ her recursion’da adım say
        steps[0] += 1
        if steps[0] > MetricsCalculator.DFS_STEP_LIMIT:
            return False

        if cur == end:
            return True

        neighbors = graph.adj.get(cur, [])
        if not neighbors:
            return False

        # True olan komşuları bul
        candidates = []
        for link in neighbors:
            nxt = link.destination

            if nxt < 0 or nxt >= len(genes):
                continue

            # Sadece TRUE node
            if not genes[nxt]:
                continue

            # Cycle yok
            if nxt in visited:
                continue

            candidates.append(nxt)

        if not candidates:
            return False

        # Rastgele sırala
        random.shuffle(candidates)

        for nxt in candidates:
            visited.add(nxt)
            path.append(nxt)

            if MetricsCalculator._dfs_true_only_random(
                graph, genes, nxt, end, visited, path, steps
            ):
                return True

            # Backtrack
            path.pop()
            visited.remove(nxt)

        return False

    # ============== 2) VALIDATION ==============

    @staticmethod
    def is_valid_path(graph, path, start, end):
        if path is None or len(path) < 2:
            return False
        if path[0] != start or path[-1] != end:
            return False

        demand = graph.get_demand(start, end)
        req_bw = demand.require_bandwidth if demand else 0

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]

            link = graph.get_link(u, v)
            if link is None:
                return False
            if req_bw > 0 and link.bandwidth < req_bw:
                return False

        return True

    # ============== 3) FITNESS HESAPLAMA ==============

    @staticmethod
    def calculate_fitness(chromosome, graph, start, end,
                          w_delay, w_reliability, w_resource):
        # 1) TRUE node'lar arasında yol bul
        path = MetricsCalculator.decode_to_path(chromosome, graph, start, end)

        # 2) Yol bulunamadıysa ceza
        if len(path) < 2:
            MetricsCalculator._punish(chromosome)
            return

        # 3) Path geçerli mi?
        if not MetricsCalculator.is_valid_path(graph, path, start, end):
            MetricsCalculator._punish(chromosome)
            return

        # 4) Geçerli path'i kaydet
        chromosome.path = list(path)

        # 5) Fitness hesapla
        total_delay = 0
        resource_cost = 0
        reliability_cost = 0

        # Link'ler üzerinden
        for i in range(len(path) - 1):
            link = graph.get_link(path[i], path[i + 1])
            if link is None:
                MetricsCalculator._punish(chromosome)
                return

            total_delay += link.link_delay
            resource_cost += (1000.0 / link.bandwidth)
            reliability_cost += -math.log(link.link_reliability)

        # Node'lar üzerinden
        for i in range(len(path)):
            node = graph.nodes.get(path[i])
            if node is None:
                MetricsCalculator._punish(chromosome)
                return

            reliability_cost += -math.log(node.reliability)

            # Ara node'ların processing delay'i
            if i > 0 and i < len(path) - 1:
                total_delay += node.proc_delay

        # Sonuçları kaydet
        chromosome.total_delay = total_delay
        chromosome.resource_cost = resource_cost
        chromosome.total_reliability = math.exp(-reliability_cost)

        # Fitness
        chromosome.fitness = (
            (w_delay * total_delay) +
            (w_reliability * reliability_cost) +
            (w_resource * resource_cost)
        )

    @staticmethod
    def calculate_fitness_from_path(chromosome, graph, start, end,
                                    w_delay, w_reliability, w_resource):
        if chromosome is None or chromosome.path is None or len(chromosome.path) < 2:
            MetricsCalculator._punish(chromosome)
            return

        path = chromosome.path

        if not MetricsCalculator.is_valid_path(graph, path, start, end):
            MetricsCalculator._punish(chromosome)
            return

        total_delay = 0
        resource_cost = 0
        reliability_cost = 0

        for i in range(len(path) - 1):
            link = graph.get_link(path[i], path[i + 1])
            if link is None:
                MetricsCalculator._punish(chromosome)
                return

            total_delay += link.link_delay
            resource_cost += (1000.0 / link.bandwidth)
            reliability_cost += -math.log(link.link_reliability)

        for i in range(len(path)):
            node = graph.nodes.get(path[i])
            if node is None:
                MetricsCalculator._punish(chromosome)
                return

            reliability_cost += -math.log(node.reliability)
            if i > 0 and i < len(path) - 1:
                total_delay += node.proc_delay

        chromosome.total_delay = total_delay
        chromosome.resource_cost = resource_cost
        chromosome.total_reliability = math.exp(-reliability_cost)

        chromosome.fitness = (
            (w_delay * total_delay) +
            (w_reliability * reliability_cost) +
            (w_resource * resource_cost)
        )

    @staticmethod
    def _punish(chromosome):
        chromosome.fitness = MetricsCalculator.PENALTY
        chromosome.total_delay = 0
        chromosome.total_reliability = 0
        chromosome.resource_cost = 0
        chromosome.path = []
