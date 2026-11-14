import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import time
import math
from datetime import datetime

from Graph import Graph
from GeneticAlgorithm import GeneticAlgorithm
from QLearning import QLearning

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="QoS Path Optimization",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 QoS Path Optimization")
st.caption("GA + Q-Learning | Feature-Complete Streamlit Version")

# ==================================================
# LOAD GRAPH
# ==================================================
@st.cache_resource
def load_graph():
    return Graph("node.csv", "link.csv", "demand.csv")

graph = load_graph()
nodes = sorted(graph.nodes.keys())

# ==================================================
# SESSION STATE
# ==================================================
defaults = {
    "ga": None,
    "ga_path": [],
    "ga_time": None,
    "ga_run_at": None,

    "ql": None,
    "ql_path": [],
    "ql_time": None,
    "ql_run_at": None,

    "start": nodes[0],
    "end": nodes[min(5, len(nodes) - 1)]
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==================================================
# SIDEBAR — GA FORM
# ==================================================
with st.sidebar.form("GA_FORM"):
    st.subheader("🧬 Genetic Algorithm")

    start = st.selectbox("Start Node", nodes, index=nodes.index(st.session_state.start))
    end = st.selectbox("End Node", nodes, index=nodes.index(st.session_state.end))

    demand_bw = st.number_input("Demand Bandwidth (Mbps)", min_value=0, step=50)

    st.markdown("### ⚖️ QoS Weights")
    w_delay = st.slider("wDelay", 0.0, 1.0, 0.4)
    w_rel   = st.slider("wReliability", 0.0, 1.0, 0.3)
    w_res   = st.slider("wResource", 0.0, 1.0, 0.3)

    ga_pop = st.slider("Population", 10, 300, 100)
    ga_gen = st.slider("Generations", 10, 300, 150)

    run_ga = st.form_submit_button("🚀 RUN GA")

# ==================================================
# SIDEBAR — QL FORM
# ==================================================
with st.sidebar.form("QL_FORM"):
    st.subheader("🤖 Q-Learning")

    episodes = st.slider("Episodes", 100, 5000, 1500)
    steps = st.slider("Max Steps", 10, 300, 120)

    alpha = st.number_input("alpha", 0.0, 1.0, 0.2)
    gamma = st.number_input("gamma", 0.0, 1.0, 0.9)
    epsilon = st.number_input("epsilon", 0.0, 1.0, 0.2)

    run_ql = st.form_submit_button("🚀 RUN QL")

# ==================================================
# RUN GA
# ==================================================
if run_ga:
    if demand_bw > 0:
        graph.set_demand(start, end, demand_bw)
    else:
        graph.remove_demand(start, end)

    st.session_state.start = start
    st.session_state.end = end

    with st.spinner("🧬 Genetic Algorithm çalışıyor..."):
        t0 = time.perf_counter()
        ga = GeneticAlgorithm(graph, start, end, w_delay, w_rel, w_res, ga_pop, ga_gen)
        best = ga.run()
        t1 = time.perf_counter()

        if best and best.path:
            st.session_state.ga = best
            st.session_state.ga_path = list(best.path)
            st.session_state.ga_time = t1 - t0
            st.session_state.ga_run_at = datetime.now()
            st.success("✅ GA tamamlandı")
        else:
            st.error("❌ GA geçerli path bulamadı")

# ==================================================
# RUN QL
# ==================================================
if run_ql:
    with st.spinner("🤖 Q-Learning çalışıyor..."):
        t0 = time.perf_counter()
        ql = QLearning(
            graph, st.session_state.start, st.session_state.end,
            w_delay, w_rel, w_res,
            episodes, steps,
            alpha, gamma, epsilon
        )
        best = ql.train_and_get_best_path()
        t1 = time.perf_counter()

        if best and best.path:
            st.session_state.ql = best
            st.session_state.ql_path = list(best.path)
            st.session_state.ql_time = t1 - t0
            st.session_state.ql_run_at = datetime.now()
            st.success("✅ Q-Learning tamamlandı")
        else:
            st.error("❌ Q-Learning path üretemedi")

# ==================================================
# NETWORK DRAW
# ==================================================
def draw_network(graph, ga_path=None, ql_path=None):
    G = nx.Graph()
    for u, links in graph.adj.items():
        for l in links:
            G.add_edge(u, l.destination)

    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()

    for u, v in G.edges():
        fig.add_trace(go.Scatter(
            x=[pos[u][0], pos[v][0], None],
            y=[pos[u][1], pos[v][1], None],
            mode="lines",
            line=dict(color="lightgray", width=1),
            showlegend=False
        ))

    if ga_path:
        for i in range(len(ga_path) - 1):
            u, v = ga_path[i], ga_path[i + 1]
            fig.add_trace(go.Scatter(
                x=[pos[u][0], pos[v][0], None],
                y=[pos[u][1], pos[v][1], None],
                mode="lines",
                line=dict(color="blue", width=4),
                name="GA"
            ))

    if ql_path:
        for i in range(len(ql_path) - 1):
            u, v = ql_path[i], ql_path[i + 1]
            fig.add_trace(go.Scatter(
                x=[pos[u][0], pos[v][0], None],
                y=[pos[u][1], pos[v][1], None],
                mode="lines",
                line=dict(color="red", width=4),
                name="QL"
            ))

    fig.add_trace(go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode="markers+text",
        marker=dict(size=14, color="gray"),
        text=[str(n) for n in G.nodes()],
        textposition="top center",
        showlegend=False
    ))

    fig.update_layout(
        height=700,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )

    return fig

# ==================================================
# MAIN VIEW
# ==================================================
st.subheader("🗺 Network Görünümü")
st.plotly_chart(
    draw_network(graph, st.session_state.ga_path, st.session_state.ql_path),
    use_container_width=True
)

# ==================================================
# BANDWIDTH + DEMAND ANALYSIS
# ==================================================
def bandwidth_analysis(path, start, end):
    if not path or len(path) < 2:
        st.info("Bandwidth analizi için path yok")
        return

    min_bw = float("inf")
    rows = []

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        link = graph.get_link(u, v)
        if link:
            bw = link.bandwidth
            min_bw = min(min_bw, bw)
            rows.append((u, v, bw))

    st.markdown("#### 📡 BANDWIDTH ANALİZİ")
    st.write(f"**Bottleneck (Min BW):** {min_bw} Mbps")

    for u, v, bw in rows:
        mark = "⚠ BOTTLENECK" if bw == min_bw else ""
        st.write(f"• {u} → {v} : {bw} Mbps {mark}")

    demand = graph.get_demand(start, end)
    st.markdown("#### 🎯 DEMAND ANALİZİ")

    if demand is None:
        st.success("Bu start-end için demand tanımlı değil")
    else:
        st.write(f"Demand: {demand.require_bandwidth} Mbps")
        if min_bw >= demand.require_bandwidth:
            st.success("✅ Demand KARŞILANIYOR")
        else:
            st.error("❌ Demand KARŞILANMIYOR")

# ==================================================
# RESULTS
# ==================================================
st.subheader("📊 SONUÇLAR")

def show_result(title, obj, path, runtime, run_at):
    if not obj or not path:
        st.info(f"{title}: sonuç yok")
        return

    st.markdown(f"### {title}")
    st.write("Path:", " → ".join(map(str, path)))
    st.write("Hop:", len(path) - 1)
    st.write("Total Delay:", f"{obj.total_delay:.2f} ms")
    st.write("Reliability:", f"{obj.total_reliability:.6f}")
    st.write("Resource Cost:", f"{obj.resource_cost:.4f}")

    # 🔥 SADECE QL FITNESS EKLENDİ
    if hasattr(obj, "fitness"):
        st.write("Fitness:", f"{obj.fitness:.4f}")
    else:
        ql_fitness = (
            w_delay * obj.total_delay +
            w_rel * (-math.log(max(obj.total_reliability, 1e-10))) +
            w_res * obj.resource_cost
        )
        st.write("Fitness:", f"{ql_fitness:.4f}")

    st.write("⏱ Çalışma Süresi:", f"{runtime:.4f} saniye")
    st.write("🕒 Çalışma Zamanı:", run_at.strftime("%d.%m.%Y %H:%M:%S"))

    bandwidth_analysis(path, st.session_state.start, st.session_state.end)

show_result(
    "🧬 Genetic Algorithm",
    st.session_state.ga,
    st.session_state.ga_path,
    st.session_state.ga_time,
    st.session_state.ga_run_at
)

show_result(
    "🤖 Q-Learning",
    st.session_state.ql,
    st.session_state.ql_path,
    st.session_state.ql_time,
    st.session_state.ql_run_at
)

st.markdown("---")
st.caption("⬇️ Scroll aktif | Bottleneck + Demand | QL Fitness eklendi")
