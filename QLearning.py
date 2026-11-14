import random
import math


class QLearning:
    class Result:
        def __init__(self, path, valid, total_delay, total_reliability, resource_cost, bottleneck):
            self.path = path
            self.valid = valid
            self.total_delay = total_delay
            self.total_reliability = total_reliability
            self.resource_cost = resource_cost
            self.bottleneck = bottleneck

    def __init__(self, graph, start, end, w_delay, w_rel, w_res, episodes, max_steps, alpha, gamma, epsilon):
        self.g = graph
        self.start = start
        self.end = end

        self.w_d = w_delay
        self.w_r = w_rel
        self.w_s = w_res

        self.Q = {}  # Q-table: {state: {action: Q_value}}

        self.episodes = episodes
        self.max_steps = max(5, max_steps)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Demand bandwidth
        demand = self.g.get_demand(start, end)
        self.req_bw = demand.require_bandwidth if demand else 0

        # En iyi path tracking
        self.best_path_so_far = []
        self.best_reward_so_far = -float('inf')

        self.init_q()

    def init_q(self):
        for s in self.g.nodes.keys():
            neighbors = self.g.adj.get(s, [])

            row = {}
            for link in neighbors:
                # Sadece BW yeterli linkleri ekle
                if self.req_bw > 0 and link.bandwidth < self.req_bw:
                    continue
                row[link.destination] = 0.0

            if row:
                self.Q[s] = row

    def train_and_get_best_path(self):
        print("🤖 Q-Learning başlıyor...")
        print(f"Episodes: {self.episodes}, MaxSteps: {self.max_steps}")
        print(f"Required BW: {self.req_bw} Mbps\n")

        success_count = 0
        stuck_count = 0

        # TRAINING LOOP
        for ep in range(self.episodes):
            cur = self.start
            visited = {cur}
            episode_path = [cur]
            episode_reward = 0
            reached_goal = False

            for step in range(self.max_steps):
                if cur == self.end:
                    reached_goal = True
                    break

                # Epsilon-greedy action seçimi
                action = self.choose_action(cur, visited)
                if action is None:
                    stuck_count += 1
                    break

                nxt = action

                # Reward hesapla
                r = self.compute_reward(cur, nxt, visited)
                episode_reward += r

                # Q-update
                old_q = self.get_q(cur, nxt)
                max_next_q = self.max_q(nxt)
                new_q = (1 - self.alpha) * old_q + self.alpha * (r + self.gamma * max_next_q)
                self.set_q(cur, nxt, new_q)

                cur = nxt
                visited.add(cur)
                episode_path.append(cur)

            if reached_goal:
                success_count += 1

                # En iyi path'i kaydet
                if episode_reward > self.best_reward_so_far:
                    self.best_reward_so_far = episode_reward
                    self.best_path_so_far = list(episode_path)

            # Epsilon decay
            if ep > self.episodes // 10:
                self.epsilon = max(0.01, self.epsilon * 0.995)

            # Progress log
            if (ep + 1) % 100 == 0:
                success_rate = (100.0 * success_count) / (ep + 1)
                print(f"Episode {ep + 1}/{self.episodes} | Success: {success_rate:.1f}% | "
                      f"Epsilon: {self.epsilon:.3f} | Best Reward: {self.best_reward_so_far:.2f}")

            # Early stopping
            if success_count > self.episodes // 5 and self.best_reward_so_far > 100:
                print("\n✅ Erken durma: Yeterince iyi çözüm bulundu!")
                break

        print("\n" + "=" * 40)
        print("✅ Eğitim tamamlandı!")
        print(f"Başarı: {success_count}/{self.episodes} ({100.0 * success_count / self.episodes:.1f}%)")
        print(f"Takılma: {stuck_count}")
        print("=" * 40 + "\n")

        # DERIVE BEST PATH
        if self.best_path_so_far and self.is_valid_path(self.best_path_so_far):
            final_path = self.best_path_so_far
            print("✅ Eğitim sırasında bulunan en iyi path kullanılıyor")
        else:
            final_path = self.greedy_path_from_q()
            print("⚠️ Greedy Q-table path kullanılıyor")

        # VALIDATE + METRICS
        valid = self.is_valid_path(final_path)
        metrics = self.compute_metrics(final_path)

        return QLearning.Result(
            final_path, valid, metrics['total_delay'],
            metrics['total_reliability'], metrics['resource_cost'],
            metrics['bottleneck']
        )

    def choose_action(self, state, visited):
        row = self.Q.get(state, {})
        if not row:
            return None

        # Sadece ziyaret edilmemiş ve BW yeterli actionlar
        valid_actions = []
        for action in row.keys():
            if action in visited:
                continue

            link = self.g.get_link(state, action)
            if link is None:
                continue
            if self.req_bw > 0 and link.bandwidth < self.req_bw:
                continue

            valid_actions.append(action)

        if not valid_actions:
            return None

        # Explore
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Exploit
        best_q = -float('inf')
        best_action = None

        for action in valid_actions:
            q = self.get_q(state, action)
            if q > best_q:
                best_q = q
                best_action = action

        return best_action if best_action else valid_actions[0]

    def compute_reward(self, u, v, visited):
        link = self.g.get_link(u, v)
        if link is None:
            return -500

        # Loop cezası
        if v in visited:
            return -300

        # BW yetersizse ceza
        if self.req_bw > 0 and link.bandwidth < self.req_bw:
            return -400

        # QoS maliyeti
        delay_cost = link.link_delay
        resource_cost = 1000.0 / link.bandwidth
        rel_cost = -math.log(max(link.link_reliability, 1e-10))

        node_v = self.g.nodes.get(v)
        if node_v:
            rel_cost += -math.log(max(node_v.reliability, 1e-10))
            if v != self.end and v != self.start:
                delay_cost += node_v.proc_delay

        qos_cost = (self.w_d * delay_cost) + (self.w_r * rel_cost) + (self.w_s * resource_cost)

        # Hedefe ulaşma bonusu
        if v == self.end:
            return 500 - qos_cost

        return -qos_cost - 1

    def get_q(self, s, a):
        return self.Q.get(s, {}).get(a, 0.0)

    def set_q(self, s, a, val):
        if s not in self.Q:
            self.Q[s] = {}
        self.Q[s][a] = val

    def max_q(self, s):
        row = self.Q.get(s, {})
        if not row:
            return 0.0
        return max(row.values())

    def greedy_path_from_q(self):
        path = [self.start]
        cur = self.start
        visited = {cur}

        for step in range(self.max_steps):
            if cur == self.end:
                break

            best_action = self.best_valid_action(cur, visited)
            if best_action is None:
                break

            path.append(best_action)
            visited.add(best_action)
            cur = best_action

        return path

    def best_valid_action(self, state, visited):
        row = self.Q.get(state, {})
        if not row:
            return None

        best_q = -float('inf')
        best_action = None

        for action, q_val in row.items():
            if action in visited:
                continue

            link = self.g.get_link(state, action)
            if link is None:
                continue
            if self.req_bw > 0 and link.bandwidth < self.req_bw:
                continue

            if q_val > best_q:
                best_q = q_val
                best_action = action

        return best_action

    def is_valid_path(self, path):
        if not path or len(path) < 2:
            return False
        if path[0] != self.start or path[-1] != self.end:
            return False

        # Cycle kontrolü
        seen = set()
        for node in path:
            if node in seen:
                return False
            seen.add(node)

        # Link ve BW kontrolü
        for i in range(len(path) - 1):
            link = self.g.get_link(path[i], path[i + 1])
            if link is None:
                return False
            if self.req_bw > 0 and link.bandwidth < self.req_bw:
                return False

        return True

    def compute_metrics(self, path):
        metrics = {
            'total_delay': 0,
            'total_reliability': 0,
            'resource_cost': 0,
            'bottleneck': float('inf')
        }

        if not path or len(path) < 2:
            return metrics

        reliability_cost = 0

        # Link'ler üzerinden
        for i in range(len(path) - 1):
            link = self.g.get_link(path[i], path[i + 1])
            if link is None:
                continue

            metrics['total_delay'] += link.link_delay
            metrics['resource_cost'] += (1000.0 / link.bandwidth)
            reliability_cost += -math.log(max(link.link_reliability, 1e-10))
            metrics['bottleneck'] = min(metrics['bottleneck'], link.bandwidth)

        # Node'lar üzerinden
        for i in range(len(path)):
            node = self.g.nodes.get(path[i])
            if node is None:
                continue

            reliability_cost += -math.log(max(node.reliability, 1e-10))
            if i > 0 and i < len(path) - 1:
                metrics['total_delay'] += node.proc_delay

        metrics['total_reliability'] = math.exp(-reliability_cost)

        if metrics['bottleneck'] == float('inf'):
            metrics['bottleneck'] = 0

        return metrics