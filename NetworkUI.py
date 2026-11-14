import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import random
import threading

from Graph import Graph
from GeneticAlgorithm import GeneticAlgorithm
from QLearning import QLearning


class NetworkUI:
    def __init__(self, root):
        self.root = root
        self.root.title("QoS Path Optimization - GA + QLearning")
        self.root.geometry("1600x900")

        # Graf yükle
        try:
            self.graph = Graph("node.csv", "link.csv", "demand.csv")
            print("✅ Graf başarıyla yüklendi!")
        except Exception as e:
            messagebox.showerror("HATA", f"Dosyalar okunamadı!\n{e}")
            self.root.quit()
            return

        # Sonuçlar
        self.best_ga = None
        self.best_ga_path = []
        self.best_ql = None
        self.best_ql_path = []

        self.last_start = 0
        self.last_end = 1

        # Node pozisyonları (canvas için)
        self.node_positions = {}
        self.generate_node_positions()

        # UI oluştur
        self.build_ui()

    def build_ui(self):
        # Ana container (scrollable)
        main_canvas = tk.Canvas(self.root, borderwidth=0)
        main_scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )

        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=main_scrollbar.set)

        main_canvas.pack(side="left", fill="both", expand=True)
        main_scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        def on_mousewheel(event):
            main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        main_canvas.bind_all("<MouseWheel>", on_mousewheel)

        # Control panel (üst)
        control_frame = ttk.LabelFrame(scrollable_frame, text="GA + QLearning Ayarları", padding=10)
        control_frame.pack(fill=tk.BOTH, padx=10, pady=10)

        # Canvas (orta)
        canvas_frame = ttk.LabelFrame(scrollable_frame, text="Network Görünümü", padding=10)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(canvas_frame, width=1400, height=600, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Result area (alt)
        result_frame = ttk.LabelFrame(scrollable_frame, text="Sonuçlar", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.txt_result = scrolledtext.ScrolledText(result_frame, width=180, height=25, font=('Courier', 9))
        self.txt_result.pack(fill=tk.BOTH, expand=True)

        # Control panel içeriği
        self.build_control_panel(control_frame)

        # İlk çizim
        self.draw_network()

    def build_control_panel(self, parent):
        # Grid layout için frame
        grid_frame = ttk.Frame(parent)
        grid_frame.pack(fill=tk.BOTH, expand=True)

        row = 0

        # Start / End
        ttk.Label(grid_frame, text="Start:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)

        nodes = sorted(self.graph.nodes.keys())
        self.combo_start = ttk.Combobox(grid_frame, values=nodes, width=10, state='readonly')
        self.combo_start.current(0)
        self.combo_start.grid(row=row, column=1, padx=5, pady=5)

        ttk.Label(grid_frame, text="End:").grid(row=row, column=2, sticky=tk.W, padx=5, pady=5)

        self.combo_end = ttk.Combobox(grid_frame, values=nodes, width=10, state='readonly')
        self.combo_end.current(min(5, len(nodes) - 1))
        self.combo_end.grid(row=row, column=3, padx=5, pady=5)

        ttk.Label(grid_frame, text="Demand (Mbps):").grid(row=row, column=4, sticky=tk.W, padx=5, pady=5)
        self.txt_demand = ttk.Entry(grid_frame, width=10)
        self.txt_demand.grid(row=row, column=5, padx=5, pady=5)

        row += 1

        # Demand buttons
        btn_apply_demand = ttk.Button(grid_frame, text="🎯 Demand Uygula", command=self.apply_demand)
        btn_apply_demand.grid(row=row, column=0, columnspan=3, padx=5, pady=5, sticky='ew')

        btn_clear_demand = ttk.Button(grid_frame, text="🗑 Demand Temizle", command=self.clear_demand)
        btn_clear_demand.grid(row=row, column=3, columnspan=3, padx=5, pady=5, sticky='ew')

        row += 1

        # Weights
        ttk.Label(grid_frame, text="wDelay:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.txt_w_delay = ttk.Entry(grid_frame, width=10)
        self.txt_w_delay.insert(0, "0.4")
        self.txt_w_delay.grid(row=row, column=1, padx=5, pady=5)

        ttk.Label(grid_frame, text="wRel:").grid(row=row, column=2, sticky=tk.W, padx=5, pady=5)
        self.txt_w_rel = ttk.Entry(grid_frame, width=10)
        self.txt_w_rel.insert(0, "0.3")
        self.txt_w_rel.grid(row=row, column=3, padx=5, pady=5)

        ttk.Label(grid_frame, text="wRes:").grid(row=row, column=4, sticky=tk.W, padx=5, pady=5)
        self.txt_w_res = ttk.Entry(grid_frame, width=10)
        self.txt_w_res.insert(0, "0.3")
        self.txt_w_res.grid(row=row, column=5, padx=5, pady=5)

        row += 1
        ttk.Separator(grid_frame, orient='horizontal').grid(row=row, column=0, columnspan=6, sticky='ew', pady=10)
        row += 1

        # GA Section
        ttk.Label(grid_frame, text="🧬 Genetik Algoritma", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=6, sticky=tk.W, padx=5, pady=5
        )
        row += 1

        self.lbl_pop = ttk.Label(grid_frame, text="GA Popülasyon: 100")
        self.lbl_pop.grid(row=row, column=0, columnspan=6, sticky=tk.W, padx=5, pady=5)
        row += 1

        self.slider_pop = ttk.Scale(grid_frame, from_=2, to=500, orient=tk.HORIZONTAL, length=600,
                                    command=self.update_pop_label)
        self.slider_pop.set(100)
        self.slider_pop.grid(row=row, column=0, columnspan=6, padx=5, pady=5, sticky='ew')
        row += 1

        self.lbl_gen = ttk.Label(grid_frame, text="GA Jenerasyon: 150")
        self.lbl_gen.grid(row=row, column=0, columnspan=6, sticky=tk.W, padx=5, pady=5)
        row += 1

        self.slider_gen = ttk.Scale(grid_frame, from_=1, to=500, orient=tk.HORIZONTAL, length=600,
                                    command=self.update_gen_label)
        self.slider_gen.set(150)
        self.slider_gen.grid(row=row, column=0, columnspan=6, padx=5, pady=5, sticky='ew')
        row += 1

        ttk.Separator(grid_frame, orient='horizontal').grid(row=row, column=0, columnspan=6, sticky='ew', pady=10)
        row += 1

        # QL Section
        ttk.Label(grid_frame, text="🤖 Q-Learning", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=6, sticky=tk.W, padx=5, pady=5
        )
        row += 1

        self.lbl_episodes = ttk.Label(grid_frame, text="QL Episodes: 1500")
        self.lbl_episodes.grid(row=row, column=0, columnspan=6, sticky=tk.W, padx=5, pady=5)
        row += 1

        self.slider_episodes = ttk.Scale(grid_frame, from_=100, to=10000, orient=tk.HORIZONTAL, length=600,
                                         command=self.update_episodes_label)
        self.slider_episodes.set(1500)
        self.slider_episodes.grid(row=row, column=0, columnspan=6, padx=5, pady=5, sticky='ew')
        row += 1

        self.lbl_steps = ttk.Label(grid_frame, text="QL MaxSteps: 120")
        self.lbl_steps.grid(row=row, column=0, columnspan=6, sticky=tk.W, padx=5, pady=5)
        row += 1

        self.slider_steps = ttk.Scale(grid_frame, from_=10, to=500, orient=tk.HORIZONTAL, length=600,
                                      command=self.update_steps_label)
        self.slider_steps.set(120)
        self.slider_steps.grid(row=row, column=0, columnspan=6, padx=5, pady=5, sticky='ew')
        row += 1

        # QL params
        ttk.Label(grid_frame, text="alpha:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=5)
        self.txt_alpha = ttk.Entry(grid_frame, width=10)
        self.txt_alpha.insert(0, "0.2")
        self.txt_alpha.grid(row=row, column=1, padx=5, pady=5)

        ttk.Label(grid_frame, text="gamma:").grid(row=row, column=2, sticky=tk.W, padx=5, pady=5)
        self.txt_gamma = ttk.Entry(grid_frame, width=10)
        self.txt_gamma.insert(0, "0.9")
        self.txt_gamma.grid(row=row, column=3, padx=5, pady=5)

        ttk.Label(grid_frame, text="epsilon:").grid(row=row, column=4, sticky=tk.W, padx=5, pady=5)
        self.txt_epsilon = ttk.Entry(grid_frame, width=10)
        self.txt_epsilon.insert(0, "0.2")
        self.txt_epsilon.grid(row=row, column=5, padx=5, pady=5)

        row += 1
        ttk.Separator(grid_frame, orient='horizontal').grid(row=row, column=0, columnspan=6, sticky='ew', pady=10)
        row += 1

        # Buttons
        self.btn_run_ga = ttk.Button(grid_frame, text="🧬 GA ÇALIŞTIR", command=self.run_ga)
        self.btn_run_ga.grid(row=row, column=0, columnspan=2, padx=5, pady=5, sticky='ew')

        self.btn_run_ql = ttk.Button(grid_frame, text="🤖 Q-LEARNING ÇALIŞTIR", command=self.run_ql)
        self.btn_run_ql.grid(row=row, column=2, columnspan=2, padx=5, pady=5, sticky='ew')

        self.btn_clear = ttk.Button(grid_frame, text="🧹 TEMİZLE", command=self.clear_results)
        self.btn_clear.grid(row=row, column=4, columnspan=2, padx=5, pady=5, sticky='ew')

    # ---------------- UI Helpers ----------------

    def set_buttons_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.btn_run_ga.config(state=state)
        self.btn_run_ql.config(state=state)
        self.btn_clear.config(state=state)

    # Slider callbacks
    def update_pop_label(self, val):
        self.lbl_pop.config(text=f"GA Popülasyon: {int(float(val))}")

    def update_gen_label(self, val):
        self.lbl_gen.config(text=f"GA Jenerasyon: {int(float(val))}")

    def update_episodes_label(self, val):
        self.lbl_episodes.config(text=f"QL Episodes: {int(float(val))}")

    def update_steps_label(self, val):
        self.lbl_steps.config(text=f"QL MaxSteps: {int(float(val))}")

    # Demand functions
    def apply_demand(self):
        start = int(self.combo_start.get())
        end = int(self.combo_end.get())
        demand_str = self.txt_demand.get().strip()

        if not demand_str:
            messagebox.showwarning("Uyarı", "Demand boş. Bir değer gir (örn: 250).")
            return

        try:
            bw = int(demand_str)
            if bw <= 0:
                raise ValueError()
            self.graph.set_demand(start, end, bw)
            messagebox.showinfo("Başarılı", f"✅ Demand uygulandı: ({start} → {end}) = {bw} Mbps")
        except ValueError:
            messagebox.showerror("Hata", "❌ Demand sayı olmalı ve > 0 olmalı (örn: 250).")

    def clear_demand(self):
        start = int(self.combo_start.get())
        end = int(self.combo_end.get())
        self.graph.remove_demand(start, end)
        messagebox.showinfo("Başarılı", f"🗑 Demand temizlendi: ({start} → {end})")

    def clear_results(self):
        self.best_ga = None
        self.best_ga_path = []
        self.best_ql = None
        self.best_ql_path = []
        self.txt_result.delete(1.0, tk.END)
        self.draw_network()

    # ---------------- Thread'li Çalıştırmalar ----------------

    def run_ga(self):
        start = int(self.combo_start.get())
        end = int(self.combo_end.get())

        # Demand otomatik uygula
        demand_str = self.txt_demand.get().strip()
        if demand_str:
            try:
                bw = int(demand_str)
                if bw > 0:
                    self.graph.set_demand(start, end, bw)
            except:
                pass

        pop = int(self.slider_pop.get())
        gen = int(self.slider_gen.get())

        w_delay = float(self.txt_w_delay.get())
        w_rel = float(self.txt_w_rel.get())
        w_res = float(self.txt_w_res.get())

        self.last_start = start
        self.last_end = end

        self.txt_result.delete(1.0, tk.END)
        self.txt_result.insert(tk.END, "⏳ GA çalışıyor...\n\n")
        self.root.update()

        # UI donmasın diye butonları kilitle
        self.set_buttons_enabled(False)

        def worker():
            try:
                ga = GeneticAlgorithm(self.graph, start, end, w_delay, w_rel, w_res, pop, gen)
                best = ga.run()
                err = None
            except Exception as e:
                best = None
                err = e

            def on_done():
                if err is not None:
                    messagebox.showerror("GA Hatası", f"GA çalışırken hata:\n{err}")
                    self.best_ga = None
                    self.best_ga_path = []
                else:
                    self.best_ga = best
                    self.best_ga_path = list(best.path) if best and best.path else []

                self.display_results(start, end)
                self.draw_network()
                self.set_buttons_enabled(True)

            self.root.after(0, on_done)

        threading.Thread(target=worker, daemon=True).start()

    def run_ql(self):
        start = int(self.combo_start.get())
        end = int(self.combo_end.get())

        # Demand otomatik uygula
        demand_str = self.txt_demand.get().strip()
        if demand_str:
            try:
                bw = int(demand_str)
                if bw > 0:
                    self.graph.set_demand(start, end, bw)
            except:
                pass

        w_delay = float(self.txt_w_delay.get())
        w_rel = float(self.txt_w_rel.get())
        w_res = float(self.txt_w_res.get())

        episodes = int(self.slider_episodes.get())
        max_steps = int(self.slider_steps.get())

        alpha = float(self.txt_alpha.get())
        gamma = float(self.txt_gamma.get())
        eps = float(self.txt_epsilon.get())

        self.last_start = start
        self.last_end = end

        self.txt_result.delete(1.0, tk.END)
        self.txt_result.insert(tk.END, "⏳ Q-Learning çalışıyor...\n\n")
        self.root.update()

        self.set_buttons_enabled(False)

        def worker():
            try:
                ql = QLearning(self.graph, start, end, w_delay, w_rel, w_res, episodes, max_steps, alpha, gamma, eps)
                best = ql.train_and_get_best_path()
                err = None
            except Exception as e:
                best = None
                err = e

            def on_done():
                if err is not None:
                    messagebox.showerror("QL Hatası", f"Q-Learning çalışırken hata:\n{err}")
                    self.best_ql = None
                    self.best_ql_path = []
                else:
                    self.best_ql = best
                    self.best_ql_path = list(best.path) if best and best.path else []

                self.display_results(start, end)
                self.draw_network()
                self.set_buttons_enabled(True)

            self.root.after(0, on_done)

        threading.Thread(target=worker, daemon=True).start()

    # ---------------- Sonuç Yazdırma + Çizim ----------------

    def display_results(self, start, end):
        self.txt_result.delete(1.0, tk.END)

        demand = self.graph.get_demand(start, end)

        result = "=" * 100 + "\n"
        result += "   QoS PATH OPTIMIZATION (GA + QLEARNING)\n"
        result += "=" * 100 + "\n\n"

        result += f"Start  : {start}\n"
        result += f"End    : {end}\n"
        result += f"Demand : {demand.require_bandwidth if demand else '(tanımlı değil)'} Mbps\n\n"

        result += "=" * 60 + " GA SONUCU " + "=" * 60 + "\n"

        if self.best_ga is None:
            result += "❌ GA daha çalıştırılmadı / sonuç yok.\n\n"
        elif not self.best_ga_path:
            result += "Path   : ❌ (Decode başarısız veya Demand/BW sağlanmadı)\n"
            result += f"FITNESS: {self.best_ga.fitness:.4f}\n\n"
        else:
            result += f"Path   : {' → '.join(map(str, self.best_ga_path))}\n"
            result += f"Hop    : {len(self.best_ga_path) - 1}\n"
            result += f"Total Delay       : {self.best_ga.total_delay:.2f} ms\n"
            result += f"Total Reliability : {self.best_ga.total_reliability:.6f}\n"
            result += f"Resource Cost     : {self.best_ga.resource_cost:.4f}\n"
            result += f"FITNESS SCORE     : {self.best_ga.fitness:.4f} ⭐\n"
            result += self.format_bw_demand_analysis(self.best_ga_path, start, end)
            result += "\n"

        result += "=" * 60 + " Q-LEARNING SONUCU " + "=" * 60 + "\n"

        if self.best_ql is None:
            result += "❌ Q-Learning daha çalıştırılmadı / sonuç yok.\n\n"
        elif not self.best_ql_path:
            result += "Path   : ❌ (QL path üretmedi)\n\n"
        else:
            result += f"Path   : {' → '.join(map(str, self.best_ql_path))}\n"
            result += f"Hop    : {len(self.best_ql_path) - 1}\n"
            result += f"Valid  : {'✅ YES' if self.best_ql.valid else '❌ NO'}\n"
            result += f"Total Delay       : {self.best_ql.total_delay:.2f} ms\n"
            result += f"Total Reliability : {self.best_ql.total_reliability:.6f}\n"
            result += f"Resource Cost     : {self.best_ql.resource_cost:.4f}\n"
            result += self.format_bw_demand_analysis(self.best_ql_path, start, end)
            result += "\n"

        self.txt_result.insert(tk.END, result)

    def format_bw_demand_analysis(self, path, start, end):
        if not path or len(path) < 2:
            return ""

        min_bw = float('inf')
        for i in range(len(path) - 1):
            link = self.graph.get_link(path[i], path[i + 1])
            if link:
                min_bw = min(min_bw, link.bandwidth)

        if min_bw == float('inf'):
            min_bw = 0

        result = "\n" + "=" * 40 + " BANDWIDTH ANALİZİ " + "=" * 40 + "\n"
        result += f"Minimum Bandwidth (Bottleneck): {min_bw if min_bw > 0 else '-'} Mbps\n\n"

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            link = self.graph.get_link(u, v)
            if link:
                mark = " ⚠ BOTTLENECK" if link.bandwidth == min_bw else ""
                result += f"• {u} → {v} : {link.bandwidth} Mbps{mark}\n"

        result += "\n" + "=" * 40 + " DEMAND ANALİZİ " + "=" * 40 + "\n"
        demand = self.graph.get_demand(start, end)

        if demand is None:
            result += "✅ Bu start-end için tanımlı demand YOK.\n"
        else:
            result += f"Demand Gereksinimi : {demand.require_bandwidth} Mbps\n"
            if min_bw >= demand.require_bandwidth:
                result += "✅ Demand KARŞILANIYOR\n"
            else:
                result += "❌ Demand KARŞILANMIYOR!\n"

        return result

    # Node positions
    def generate_node_positions(self):
        random.seed(42)
        for node_id in self.graph.nodes.keys():
            x = 50 + random.randint(0, 1300)
            y = 50 + random.randint(0, 500)
            self.node_positions[node_id] = (x, y)

    # Draw network
    def draw_network(self):
        self.canvas.delete("all")

        # Legend
        self.canvas.create_text(
            10, 10, anchor=tk.NW,
            text="Gri: Linkler | Mavi: GA yolu | Kırmızı: QL yolu | Yeşil: Start | Turuncu: End",
            font=('Arial', 9, 'bold')
        )

        # Draw links
        for u, neighbors in self.graph.adj.items():
            pu = self.node_positions.get(u)
            if not pu:
                continue

            for link in neighbors:
                pv = self.node_positions.get(link.destination)
                if pv:
                    self.canvas.create_line(pu[0], pu[1], pv[0], pv[1], fill='lightgray', width=1)

        # Draw GA path
        if self.best_ga_path and len(self.best_ga_path) >= 2:
            for i in range(len(self.best_ga_path) - 1):
                a = self.node_positions.get(self.best_ga_path[i])
                b = self.node_positions.get(self.best_ga_path[i + 1])
                if a and b:
                    self.canvas.create_line(a[0], a[1], b[0], b[1], fill='blue', width=3)

        # Draw QL path
        if self.best_ql_path and len(self.best_ql_path) >= 2:
            for i in range(len(self.best_ql_path) - 1):
                a = self.node_positions.get(self.best_ql_path[i])
                b = self.node_positions.get(self.best_ql_path[i + 1])
                if a and b:
                    self.canvas.create_line(a[0], a[1], b[0], b[1], fill='red', width=3)

        # Draw nodes
        for node_id, pos in self.node_positions.items():
            x, y = pos

            # Node color
            color = 'gray'
            if self.best_ga_path and node_id in self.best_ga_path:
                color = 'blue'
            if self.best_ql_path and node_id in self.best_ql_path:
                color = 'red'

            if node_id == self.last_start:
                color = 'green'
            if node_id == self.last_end:
                color = 'orange'

            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill=color, outline='black', width=2)
            self.canvas.create_text(x + 15, y, text=str(node_id), font=('Arial', 9, 'bold'))


def main():
    root = tk.Tk()
    app = NetworkUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
