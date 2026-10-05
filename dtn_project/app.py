"""Streamlit dashboard: AI-Optimized Delay Tolerant Network for Disaster Communication."""
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import database as db
from ai_router import FEATURE_LABELS, FEATURES, AIRouter
from dtn import PRIORITIES, PROTOCOLS, Simulation, compare_protocols

st.set_page_config(page_title="AI-Optimized DTN", page_icon="📡", layout="wide")
db.init_db()

st.markdown(
    """
    <style>
    .banner {background: linear-gradient(90deg,#0b3d91,#1b7f9e); padding: 18px 24px;
             border-radius: 10px; color: white; margin-bottom: 12px;}
    .banner h2 {margin: 0; color: white;}
    .banner p {margin: 4px 0 0 0; opacity: .92;}
    div[data-testid="stMetric"] {background:#f4f7fb; border:1px solid #e1e8f0;
             padding:8px 12px; border-radius:8px;}
    </style>
    <div class="banner">
      <h2>📡 AI-Optimized Delay Tolerant Network for Disaster Communication</h2>
      <p>STORE → CARRY → FORWARD &nbsp;|&nbsp; User → Node → Rescue Vehicle → Node → Hospital
         &nbsp;|&nbsp; Simulation prototype (no real Bluetooth/Wi-Fi)</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------- helpers
def show_plot(container, fig):
    """plotly_chart wrapper that works across Streamlit versions."""
    st.session_state["_k"] = st.session_state.get("_k", 0) + 1
    key = f"plot_{st.session_state['_k']}"
    try:
        container.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        container.plotly_chart(fig, use_container_width=True, key=key)


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kw)


@st.cache_resource(show_spinner="Training Random Forest on simulated data...")
def get_ai():
    return AIRouter().train()


ai = get_ai()


# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Simulation settings")
    n_nodes = st.slider("Number of nodes", 10, 20, 15)
    comm_range = st.slider("Communication range (units)", 8, 30, 14)
    buffer_size = st.slider("User buffer capacity (messages)", 3, 15, 8)
    auto_rate = st.slider("Auto emergency-message rate (per min)", 0.0, 0.5, 0.12, 0.01)
    default_ttl = st.slider("Default TTL (minutes)", 60, 600, 300, 10)
    seed = st.number_input("Random seed", 0, 9999, 42)
    protocol = st.selectbox("Routing protocol (live simulation)", PROTOCOLS, index=2)
    show_range = st.checkbox("Show communication range circles", value=False)
    st.divider()
    st.subheader("▶ Playback")
    anim_steps = st.slider("Steps to animate", 10, 300, 60, 10)
    anim_delay = st.slider("Frame delay (sec)", 0.0, 0.5, 0.05, 0.05)
    apply_reset = st.button("🔄 Apply settings & reset")
    st.caption("1 step = 1 simulated minute. Protocol changes apply instantly; other settings need a reset.")


def build_sim():
    return Simulation(protocol=protocol, n_nodes=n_nodes, comm_range=comm_range, seed=int(seed),
                      buffer_size=buffer_size, ai=ai, auto_rate=auto_rate, default_ttl=default_ttl)


if "sim" not in st.session_state or apply_reset:
    st.session_state.sim = build_sim()
    st.session_state.selected = None
sim: Simulation = st.session_state.sim
sim.protocol = protocol


# ----------------------------------------------------------------- map & KPIs
def make_map(sim: Simulation, selected_id, show_range):
    fig = go.Figure()
    A = sim.area
    xs, ys = [], []
    for i, j in sim.contacts:
        xs += [sim.nodes[i].x, sim.nodes[j].x, None]
        ys += [sim.nodes[i].y, sim.nodes[j].y, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Active contact",
                             line=dict(color="rgba(110,110,110,0.6)", width=1.5, dash="dot"),
                             hoverinfo="skip"))

    msg = sim.messages.get(selected_id) if selected_id else None
    if msg is not None:
        pts = list(msg.path_points)
        if msg.holder is not None:
            pts.append((sim.nodes[msg.holder].x, sim.nodes[msg.holder].y))
        if len(pts) > 1:
            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts],
                                     mode="lines+markers", name=f"Path of {msg.msg_id}",
                                     line=dict(color="#7b2cbf", width=3, dash="dash"),
                                     marker=dict(size=6)))

    styles = {"user": ("Survivor / user node", "circle", "#2b7bd6", 13),
              "vehicle": ("Rescue vehicle", "diamond", "#f28c28", 17),
              "hospital": ("Hospital", "square", "#2e9e5b", 22)}
    for kind, (label, symbol, color, size) in styles.items():
        ns = [n for n in sim.nodes if n.kind == kind]
        hover = [f"<b>{n.name}</b><br>Battery: {n.battery:.1f}%<br>Buffer: {len(n.buffer)}/{n.capacity}"
                 f"<br>Encounters: {n.encounters}<br>Delivery rate: {n.delivery_rate:.2f}"
                 f"<br>Pos: ({n.x:.0f}, {n.y:.0f})" for n in ns]
        fig.add_trace(go.Scatter(
            x=[n.x for n in ns], y=[n.y for n in ns], mode="markers+text", name=label,
            text=[n.short for n in ns], textposition="top center", hovertext=hover, hoverinfo="text",
            marker=dict(symbol=symbol, size=size, color=[color if n.alive else "#999999" for n in ns],
                        line=dict(width=1.5, color="white"))))

    carriers = [n for n in sim.nodes if n.buffer]
    if carriers:
        fig.add_trace(go.Scatter(
            x=[n.x for n in carriers], y=[n.y for n in carriers], mode="markers",
            name="Carrying messages (STORE/CARRY)", hoverinfo="text",
            hovertext=[f"{n.name} carries: " + ", ".join(n.buffer.keys()) for n in carriers],
            marker=dict(symbol="circle-open", size=30, color="#7b2cbf", line=dict(width=2.5))))

    if msg is not None:
        src, dst = sim.nodes[msg.source], sim.nodes[msg.destination]
        fig.add_trace(go.Scatter(x=[src.x], y=[src.y], mode="markers+text", text=["SOURCE"],
                                 textposition="bottom center", name="Selected source",
                                 marker=dict(symbol="star", size=24, color="#1d4ed8",
                                             line=dict(width=1, color="white"))))
        fig.add_trace(go.Scatter(x=[dst.x], y=[dst.y], mode="markers+text", text=["DESTINATION"],
                                 textposition="bottom center", name="Selected destination",
                                 marker=dict(symbol="star", size=24, color="#15803d",
                                             line=dict(width=1, color="white"))))
        if msg.holder is not None:
            h = sim.nodes[msg.holder]
            fig.add_trace(go.Scatter(x=[h.x], y=[h.y], mode="markers+text", text=["CARRIER"],
                                     textposition="bottom center", name="Current carrier",
                                     marker=dict(symbol="star-triangle-up", size=22, color="#dc2626",
                                                 line=dict(width=1, color="white"))))

    la = sim.last_ai
    if la and sim.t - la["step"] <= 15:
        n = sim.nodes[la["to"]]
        fig.add_trace(go.Scatter(
            x=[n.x], y=[n.y], mode="markers+text", name="AI-selected relay",
            text=[f"AI pick p={la['prob']:.2f}"], textposition="top center",
            marker=dict(symbol="circle-open", size=40, color="#e6b800", line=dict(width=4))))

    recent = [e for e in sim.events if e["type"] in ("FORWARD", "DELIVERED")
              and e["step"] >= sim.t - 10 and e["fx"] is not None][-14:]
    annotations = []
    for e in recent:
        color = "#16a34a" if e["type"] == "DELIVERED" else ("#dc2626" if e["prob"] is not None else "#f97316")
        annotations.append(dict(x=e["tx"], y=e["ty"], ax=e["fx"], ay=e["fy"], xref="x", yref="y",
                                axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.3,
                                arrowwidth=2.5, arrowcolor=color, opacity=0.85))
    shapes = []
    if show_range:
        for n in sim.nodes:
            r = sim.comm_range
            shapes.append(dict(type="circle", xref="x", yref="y", x0=n.x - r, x1=n.x + r,
                               y0=n.y - r, y1=n.y + r, line=dict(color="rgba(80,120,200,0.25)", width=1)))
    fig.update_layout(
        height=640, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="#f7f9fc",
        title=f"Disaster area — t = {sim.t} min | protocol: {sim.protocol} "
              f"(green arrow = delivered, red = AI forward, orange = random forward)",
        xaxis=dict(range=[-3, A + 3], showgrid=True, gridcolor="#e5eaf1", zeroline=False),
        yaxis=dict(range=[-3, A + 3], scaleanchor="x", showgrid=True, gridcolor="#e5eaf1", zeroline=False),
        legend=dict(orientation="h", y=-0.05), annotations=annotations, shapes=shapes)
    return fig


def kpi_block(sim: Simulation):
    m = sim.metrics()
    c = st.columns(5)
    c[0].metric("Simulated time", f"{sim.t} min")
    c[1].metric("Total messages", m["total"])
    c[2].metric("Delivered", m["delivered"])
    c[3].metric("Pending", m["pending"])
    c[4].metric("Expired (TTL)", m["expired"])
    c = st.columns(5)
    c[0].metric("Dropped (buffer)", m["dropped"])
    c[1].metric("Delivery ratio", f"{100 * m['delivery_ratio']:.1f}%")
    c[2].metric("Average delay", "—" if np.isnan(m["avg_delay"]) else f"{m['avg_delay']:.1f} min")
    c[3].metric("Average hops", "—" if np.isnan(m["avg_hops"]) else f"{m['avg_hops']:.2f}")
    ap = m["avg_ai_prob"]
    c[4].metric("AI delivery probability", "—" if (np.isnan(ap) or sim.protocol != "AI Routing") else f"{ap:.2f}",
                help="Mean AI-estimated delivery probability of pending messages' current carriers")
    st.caption(f"Encounters so far: **{sim.total_encounters}** · Transmissions: **{m['transmissions']}** · "
               f"Duplicates blocked: **{sim.duplicates_blocked}** · Active links: **{len(sim.contacts)}**")


def redraw(map_ph, kpi_ph):
    show_plot(map_ph, make_map(sim, st.session_state.get("selected"), show_range))
    with kpi_ph.container():
        kpi_block(sim)


# ----------------------------------------------------------------- controls
b1, b2, b3, b4 = st.columns(4)
if b1.button("⏭ Step +1"):
    sim.step()
if b2.button("⏩ Step +10"):
    for _ in range(10):
        sim.step()
animate = b3.button("▶ Animate", type="primary")
if b4.button("💾 Save run to database"):
    rid = db.save_run("live", sim.protocol,
                      {"seed": int(seed), "n_nodes": n_nodes, "steps": sim.t, "comm_range": comm_range},
                      sim.metrics(), sim.messages_records(), sim.events_records())
    st.success(f"Run saved to SQLite with run_id = {rid}")

kpi_ph = st.empty()
map_ph = st.empty()
if animate:
    for i in range(anim_steps):
        sim.step()
        if i % 2 == 1 or i == anim_steps - 1:
            redraw(map_ph, kpi_ph)
            time.sleep(anim_delay)
else:
    redraw(map_ph, kpi_ph)

# ----------------------------------------------------------------- tabs
tabs = st.tabs(["✉️ Create message", "📨 Messages", "📟 Nodes", "🤖 AI router",
                "📊 Protocol comparison", "🗄️ Logs & database", "📘 About"])

# --- Create message
with tabs[0]:
    st.subheader("Create an emergency message")
    names = [n.name for n in sim.nodes]
    users = [n.name for n in sim.nodes if n.kind != "hospital"]
    with st.form("create_msg"):
        c1, c2, c3, c4 = st.columns(4)
        src = c1.selectbox("Source", users, index=0)
        dst = c2.selectbox("Destination", names, index=0)
        prio = c3.selectbox("Priority", PRIORITIES)
        ttl = c4.number_input("TTL (minutes)", 10, 1000, default_ttl)
        text = st.text_input("Message text (optional)", "Need medical help, building collapsed")
        ok = st.form_submit_button("Send into the network")
    if ok:
        if src == dst:
            st.error("Source and destination must differ.")
        else:
            m = sim.create_message(sim.node_by_name(src).id, sim.node_by_name(dst).id, prio, ttl, text)
            st.session_state.selected = m.msg_id
            st.success(f"{m.msg_id} created and STORED at {src}. It will be carried until the node meets a better "
                       f"relay or the destination. Press Step/Animate and track it in the map.")
    if st.button("🎬 Inject demo scenario (User → Rescue Vehicle → Hospital)"):
        user = next(n for n in sim.nodes if n.kind == "user")
        hosp = next(n for n in sim.nodes if n.kind == "hospital")
        m = sim.create_message(user.id, hosp.id, "HIGH", 400, "Demo: trapped survivors, need ambulance")
        st.session_state.selected = m.msg_id
        st.success(f"{m.msg_id}: {user.name} → {hosp.name} created and selected for tracking. Press Animate.")

# --- Messages
with tabs[1]:
    st.subheader("Message tracker")
    mdf = sim.messages_df()
    if mdf.empty:
        st.info("No messages yet. Step the simulation or create one.")
    else:
        ids = list(mdf["ID"])
        cur = st.session_state.get("selected")
        idx = ids.index(cur) if cur in ids else 0
        st.session_state.selected = st.selectbox("Message highlighted on the map", ids, index=idx)
        show_df(mdf)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Delivery by priority** (HIGH is forwarded first and evicted last)")
            show_df(sim.priority_stats())
        with c2:
            if sim.history:
                hist = pd.DataFrame(sim.history).melt("step", ["delivered", "pending", "expired", "dropped"],
                                                      var_name="status", value_name="messages")
                show_plot(st, px.line(hist, x="step", y="messages", color="status",
                                      title="Message status over time"))

# --- Nodes
with tabs[2]:
    st.subheader("Node status")
    show_df(sim.nodes_df())
    pick = st.selectbox("Inspect node", [n.name for n in sim.nodes])
    n = sim.node_by_name(pick)
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Buffer (stored messages)**")
    c1.write(", ".join(n.buffer.keys()) or "empty")
    c2.markdown("**Delivered to this node**")
    c2.write(", ".join(n.delivered_to_me) or "none")
    c3.markdown("**Helped deliver (delivery history)**")
    c3.write(", ".join(n.delivery_history) or "none")
    enc = pd.DataFrame([{"Met node": sim.nodes[k].name, "Encounters": v} for k, v in n.enc_with.items()])
    if not enc.empty:
        show_plot(st, px.bar(enc.sort_values("Encounters", ascending=False), x="Met node", y="Encounters",
                             title=f"Encounter count of {pick}"))

# --- AI router
with tabs[3]:
    st.subheader("Random Forest routing model")
    st.warning("The training data is SIMULATED (synthetic). No real disaster-network trace was available, so "
               "samples were generated from a noisy logistic rule over the six features. Replace it with logged "
               "simulation/real data to retrain.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Hold-out accuracy", f"{ai.accuracy:.3f}")
    c2.metric("Hold-out ROC-AUC", f"{ai.auc:.3f}")
    c3.metric("Training samples (simulated)", len(ai.dataset))
    imp = ai.importances.rename(index=FEATURE_LABELS).reset_index()
    imp.columns = ["Feature", "Importance"]
    show_plot(st, px.bar(imp, x="Importance", y="Feature", orientation="h", title="Feature importance"))
    st.markdown("**Try the model** — estimate the delivery probability of a candidate forwarding node:")
    c = st.columns(6)
    v = [c[0].slider("Distance to dest.", 0.0, 1.0, 0.3, 0.01),
         c[1].slider("Battery", 0.0, 1.0, 0.8, 0.01),
         c[2].slider("Buffer free", 0.0, 1.0, 0.6, 0.01),
         c[3].slider("Encounter freq.", 0.0, 1.0, 0.5, 0.01),
         c[4].slider("Delivery rate", 0.0, 1.0, 0.6, 0.01),
         c[5].slider("Hop count", 0, 8, 2)]
    p = ai.predict_one(*v)
    g = go.Figure(go.Indicator(mode="gauge+number", value=100 * p, number={"suffix": "%"},
                               title={"text": "Estimated delivery probability"},
                               gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1b7f9e"}}))
    g.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    show_plot(st, g)
    with st.expander("Sample of the simulated training dataset"):
        show_df(ai.dataset.head(20).round(3))

# --- Comparison
with tabs[4]:
    st.subheader("Direct Delivery vs Random Routing vs AI Routing")
    st.caption("All protocols run on identical mobility and message schedules (same seeds), using the sidebar "
               "settings for nodes, range, buffer, message rate and TTL.")
    c1, c2 = st.columns(2)
    cmp_steps = c1.slider("Simulated minutes per run", 150, 800, 400, 50)
    cmp_seeds = c2.slider("Number of random seeds (averaged)", 1, 8, 3)
    if st.button("🚀 Run comparison", type="primary"):
        bar = st.progress(0.0, text="Running simulations...")
        summary, runs, series = compare_protocols(
            ai, steps=cmp_steps, seeds=tuple(range(1, cmp_seeds + 1)), progress=lambda f: bar.progress(f),
            n_nodes=n_nodes, comm_range=comm_range, buffer_size=buffer_size, auto_rate=auto_rate,
            default_ttl=default_ttl)
        bar.empty()
        st.session_state.cmp = (summary, runs, series)
        for _, r in summary.iterrows():
            db.save_run("comparison", r["Protocol"],
                        {"seed": f"avg of {cmp_seeds}", "n_nodes": n_nodes, "steps": cmp_steps,
                         "comm_range": comm_range},
                        {"total": int(r["Total"]), "delivered": int(round(r["Delivered"])), "pending": 0,
                         "expired": 0, "dropped": 0, "delivery_ratio": r["Delivery Ratio (%)"] / 100,
                         "avg_delay": r["Avg Delay (min)"], "avg_hops": r["Avg Hops"],
                         "transmissions": int(r["Transmissions"])}, [], [])
    if "cmp" in st.session_state:
        summary, runs, series = st.session_state.cmp
        colors = {"Direct Delivery": "#9ca3af", "Random Routing": "#f97316", "AI Routing": "#1b7f9e"}
        c1, c2 = st.columns(2)
        with c1:
            f = px.bar(summary, x="Protocol", y="Delivery Ratio (%)", color="Protocol", text_auto=".1f",
                       error_y="Ratio Std", color_discrete_map=colors, title="Delivery ratio (higher is better)")
            show_plot(st, f)
        with c2:
            f = px.bar(summary, x="Protocol", y="Avg Delay (min)", color="Protocol", text_auto=".1f",
                       color_discrete_map=colors, title="Average delay of delivered messages (lower is better)")
            show_plot(st, f)
        c1, c2 = st.columns(2)
        with c1:
            f = px.line(series, x="Step", y="Delivery Ratio (%)", color="Protocol",
                        color_discrete_map=colors, title="Delivery ratio over time")
            show_plot(st, f)
        with c2:
            f = px.bar(summary, x="Protocol", y="Tx per Delivered", color="Protocol", text_auto=".1f",
                       color_discrete_map=colors, title="Network overhead: transmissions per delivered message")
            show_plot(st, f)
        show_df(summary.round(2))
        with st.expander("Per-seed results"):
            show_df(runs.round(2))
        best = summary.loc[summary["Delivery Ratio (%)"].idxmax(), "Protocol"]
        st.success(f"Best delivery ratio in this experiment: **{best}**. Avg delay is only computed over delivered "
                   f"messages, so read it together with the delivery ratio.")
    else:
        st.info("Click **Run comparison** to simulate all three protocols.")

# --- Logs & DB
with tabs[5]:
    st.subheader("Live event log")
    edf = sim.events_df()
    show_df(edf.tail(200).iloc[::-1])
    st.subheader("Saved runs (SQLite: dtn_results.db)")
    runs_df = db.load_runs()
    if runs_df.empty:
        st.info("No saved runs yet. Use 'Save run to database' above or run a comparison.")
    else:
        show_df(runs_df.drop(columns=["params"]))
        rid = st.selectbox("Inspect run", list(runs_df["run_id"]))
        c1, c2 = st.columns(2)
        c1.markdown("**Messages**")
        with c1:
            show_df(db.load_messages(rid))
        c2.markdown("**Events (first 500)**")
        with c2:
            show_df(db.load_events(rid))
    if st.button("🗑 Clear database"):
        db.clear_db()
        st.rerun()

# --- About
with tabs[6]:
    st.markdown(
        """
### Architecture
`app.py` (Streamlit UI) → `dtn.py` (mobility, encounters, buffers, TTL, routing) → `ai_router.py`
(Random Forest on synthetic data) → `database.py` (SQLite: runs, messages, events).

### How the DTN works
1. **STORE** – a message is created and kept in the source node's buffer (no end-to-end path exists).
2. **CARRY** – nodes move; the carrier keeps the message until it meets another node.
3. **FORWARD** – when two nodes are within communication range (an *encounter*) the carrier hands the
   message over if the routing protocol says so. Delivery happens when the destination is in range.
Messages expire at their TTL, duplicates are blocked by a per-node *seen* list, and when a buffer is full the
lowest-priority message is evicted. HIGH priority messages are forwarded first.

### How AI routing works
For every pending message the carrier scores itself and every neighbour in range using a Random Forest
(distance to destination, battery, buffer availability, encounter frequency, past delivery rate, hop count).
The neighbour with the highest probability is chosen, and the message is forwarded only if it beats the
carrier's own score by a margin (prevents ping-pong).

*Training data is simulated. This is a prototype, not a real Bluetooth/Wi-Fi deployment.*
        """
    )
