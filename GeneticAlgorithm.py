import random
import time
from Chromosome import Chromosome
from MetricsCalculator import MetricsCalculator


class GeneticAlgorithm:
    TIME_LIMIT_MS = 50000
    MUT_RATE = 0.15
    NODE_INCLUDE_PROB = 0.3

    def __init__(self, graph, start, end, w_delay, w_rel, w_res, pop_size, gen_count):
        self.graph = graph
        self.start = start
        self.end = end

        self.w_d = w_delay
        self.w_r = w_rel
        self.w_s = w_res

        self.POP = pop_size
        self.GEN = gen_count

        self.ELITE = max(1, pop_size // 20)
        self.TOURNAMENT = max(2, pop_size // 10)

    # ============================================================
    # INIT POPULATION (SEED + RANDOM, FITNESS HAZIR)
    # ============================================================
    '''
    def init_population_random_bits(self):
        pop = []
        n = len(self.graph.nodes)

        TARGET_SEED = self.POP // 4
        MAX_TRY = self.POP // 2

        unique_paths = set()
        tries = 0


        # -------- BW-valid SEED üret --------
        while len(pop) < TARGET_SEED and tries < MAX_TRY:
            tries += 1
            path = self.graph.find_bw_valid_path(self.start, self.end)
            if not path:
                break

            key = tuple(path)
            if key in unique_paths:
                continue
            unique_paths.add(key)

            c = Chromosome(gene_size=n)
            for node in path:
                c.genes[node] = True

            c.genes[self.start] = True
            c.genes[self.end] = True
            c.path = list(path)

            MetricsCalculator.calculate_fitness_from_path(
                c, self.graph, self.start, self.end,
                self.w_d, self.w_r, self.w_s
            )

            pop.append(c)

        print(f"✅ Seeded BW-valid individuals: {len(pop)}")

        # -------- KALANI RANDOM DOLDUR --------
        while len(pop) < self.POP:
            c = Chromosome(gene_size=n)
            for i in range(n):
                c.genes[i] = random.random() < self.NODE_INCLUDE_PROB

            c.genes[self.start] = True
            c.genes[self.end] = True

            MetricsCalculator.calculate_fitness(
                c, self.graph, self.start, self.end,
                self.w_d, self.w_r, self.w_s
            )

            pop.append(c)

        return pop
'''

    def init_population_random_bits(self):
        pop = []
        n = len(self.graph.nodes)

        TARGET_SEED = self.POP // 4  # toplam seed sayısı
        BFS_COUNT = 1  # 1 tane BFS
        DFS_COUNT = TARGET_SEED - BFS_COUNT  # geri kalanı DFS
        MAX_TRY = self.POP // 2

        unique_paths = set()

        # =================================================
        # 1️⃣ BFS ile 1 tane kaliteli seed
        # =================================================
        bfs_path = self.graph.find_bw_valid_path_bfs(self.start, self.end)
        if bfs_path:
            key = tuple(bfs_path)
            unique_paths.add(key)

            c = Chromosome(gene_size=n)
            for node in bfs_path:
                c.genes[node] = True

            c.genes[self.start] = True
            c.genes[self.end] = True
            c.path = list(bfs_path)

            MetricsCalculator.calculate_fitness_from_path(
                c, self.graph, self.start, self.end,
                self.w_d, self.w_r, self.w_s
            )

            pop.append(c)

        # =================================================
        # 2️⃣ DFS ile farklı seed'ler
        # =================================================
        tries = 0
        while len(pop) < BFS_COUNT + DFS_COUNT and tries < MAX_TRY:
            tries += 1
            path = self.graph.find_bw_valid_path(self.start, self.end)  # DFS

            if not path:
                continue

            key = tuple(path)
            if key in unique_paths:
                continue

            unique_paths.add(key)

            c = Chromosome(gene_size=n)
            for node in path:
                c.genes[node] = True

            c.genes[self.start] = True
            c.genes[self.end] = True
            c.path = list(path)

            MetricsCalculator.calculate_fitness_from_path(
                c, self.graph, self.start, self.end,
                self.w_d, self.w_r, self.w_s
            )

            pop.append(c)

        print(f"✅ Seeded individuals (BFS + DFS): {len(pop)}")

        # =================================================
        # 3️⃣ KALANI RANDOM
        # =================================================
        while len(pop) < self.POP:
            c = Chromosome(gene_size=n)

            for i in range(n):
                c.genes[i] = random.random() < self.NODE_INCLUDE_PROB

            c.genes[self.start] = True
            c.genes[self.end] = True

            MetricsCalculator.calculate_fitness(
                c, self.graph, self.start, self.end,
                self.w_d, self.w_r, self.w_s
            )

            pop.append(c)

        return pop

    # ============================================================
    # RUN
    # ============================================================
    def run(self):
        t0 = time.time() * 1000  # ms

        pop = self.init_population_random_bits()
        if not pop:
            return None

        print("===== INITIAL POPULATION =====")
        for i, c in enumerate(pop):
            print(f"INIT[{i}] fit={c.fitness:.4f} | path={c.path}")

        best = Chromosome(other=self.get_best(pop))

        # ================= EVOLUTION =================
        for gen in range(1, self.GEN + 1):

            if (time.time() * 1000 - t0) > self.TIME_LIMIT_MS:
                print("\n⏱ Zaman limiti aşıldı!")
                return best

            print(f"\n========== GEN {gen} ==========")

            new_pop = []

            # -------- ELITIZM --------
            temp = list(pop)
            for i in range(self.ELITE):
                elite = self.get_best(temp)
                if elite is None:
                    break

                new_pop.append(Chromosome(other=elite))
                temp.remove(elite)

                if i == 0:
                    print(f"ELITE[0] fit={elite.fitness:.4f} | path={elite.path}")

            # -------- ÇOCUK ÜRET --------
            while len(new_pop) < self.POP:

                if (time.time() * 1000 - t0) > self.TIME_LIMIT_MS:
                    break

                p1 = self.tournament(pop)
                p2 = self.tournament(pop)
                if p1 is None or p2 is None:
                    continue

                child = self.one_point_crossover(p1, p2)

                if random.random() < self.MUT_RATE:
                    child = self.mutate_probabilistic(child)

                child.genes[self.start] = True
                child.genes[self.end] = True

                MetricsCalculator.calculate_fitness(
                    child, self.graph, self.start, self.end,
                    self.w_d, self.w_r, self.w_s
                )

                new_pop.append(child)

            pop = new_pop

            current_best = self.get_best(pop)
            if current_best and current_best.fitness < best.fitness:
                best = Chromosome(other=current_best)

            print(f"BEST GEN {gen} => fit={best.fitness:.4f} | path={best.path}")

        return best

    # ============================================================
    # OPERATORS
    # ============================================================
    def tournament(self, pop):
        if not pop:
            return None

        shuffled = list(pop)
        random.shuffle(shuffled)

        limit = min(self.TOURNAMENT, len(shuffled))
        best = shuffled[0]

        for i in range(1, limit):
            if shuffled[i].fitness < best.fitness:
                best = shuffled[i]

        return Chromosome(other=best)

    def one_point_crossover(self, p1, p2):
        n = len(p1.genes)
        child = Chromosome(gene_size=n)

        cut = 1 + random.randint(0, n - 3)

        for i in range(n):
            child.genes[i] = p1.genes[i] if i < cut else p2.genes[i]

        child.genes[self.start] = True
        child.genes[self.end] = True
        return child

    def mutate_one_flip(self, c):
        m = Chromosome(other=c)
        n = len(m.genes)

        idx = random.randint(0, n - 1)
        guard = 0
        while (idx == self.start or idx == self.end) and guard < 30:
            idx = random.randint(0, n - 1)
            guard += 1

        if idx != self.start and idx != self.end:
            m.genes[idx] = not m.genes[idx]

        m.genes[self.start] = True
        m.genes[self.end] = True
        return m

    def mutate_probabilistic(self, c, p=0.1):
        m = Chromosome(other=c)

        for i in range(len(m.genes)):
            # start ve end sabit kalsın
            if i == self.start or i == self.end:
                continue

            # belirli olasılıkla flip
            if random.random() < p:
                m.genes[i] = not m.genes[i]

        m.genes[self.start] = True
        m.genes[self.end] = True
        return m

    def get_best(self, pop):
        if not pop:
            return None

        best = pop[0]
        for c in pop:
            if c.fitness < best.fitness:
                best = c

        return best
