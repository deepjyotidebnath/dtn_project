"""Core Delay Tolerant Network (DTN) simulation engine.

Implements: node mobility, communication range, encounter detection,
STORE -> CARRY -> FORWARD, buffering with priority eviction, TTL expiry,
duplicate detection and three routing protocols (Direct, Random, AI).

One simulation step = one simulated minute. Distances are abstract units
(roughly 100 m each) inside a square disaster area.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PRIORITIES = ["HIGH", "MEDIUM", "LOW"]
PRIORITY_RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
TTL_FACTOR = {"HIGH": 1.0, "MEDIUM": 0.8, "LOW": 0.6}
PROTOCOLS = ["Direct Delivery", "Random Routing", "AI Routing"]


@dataclass
class Message:
    msg_id: str
    source: int
    destination: int
    priority: str
    ttl: int
    created_step: int
    text: str = ""
    hops: int = 0
    status: str = "PENDING"          # PENDING / DELIVERED / EXPIRED / DROPPED
    delivered_step: Optional[int] = None
    holder: Optional[int] = None     # node currently carrying the bundle
    path: List[int] = field(default_factory=list)
    path_points: List[Tuple[float, float]] = field(default_factory=list)
    ai_prob: Optional[float] = None  # latest AI delivery probability

    @property
    def rank(self) -> int:
        return PRIORITY_RANK[self.priority]


class Node:
    def __init__(self, nid, kind, name, short, x, y, speed, capacity, battery):
        self.id = nid
        self.kind = kind             # user / vehicle / hospital
        self.name = name
        self.short = short
        self.x, self.y = x, y
        self.speed = speed
        self.capacity = capacity
        self.battery = battery       # percent
        self.buffer: Dict[str, Message] = {}
        self.seen: set = set()       # duplicate detection (bundle IDs ever held)
        self.encounters = 0
        self.enc_with: Dict[int, int] = {}
        self.relay_carried = 0       # bundles this node has carried
        self.relay_success = 0       # carried bundles that were later delivered
        self.delivered_to_me: List[str] = []   # bundles that arrived here as final destination
        self.delivery_history: List[str] = []  # bundles this node helped deliver
        self.tx = 0
        self.rx = 0
        self.target = (x, y)

    @property
    def alive(self) -> bool:
        return self.battery > 0

    @property
    def delivery_rate(self) -> float:
        # Laplace-smoothed success rate so new nodes start at 0.5
        return (self.relay_success + 1) / (self.relay_carried + 2)


class Simulation:
    def __init__(self, protocol="AI Routing", n_nodes=15, area=100.0, comm_range=14.0,
                 seed=42, buffer_size=8, ai=None, auto_rate=0.12, default_ttl=300,
                 max_transfers=3, gen_until=None, ai_margin=0.05):
        assert protocol in PROTOCOLS
        self.protocol = protocol
        self.area = float(area)
        self.diag = math.hypot(area, area)
        self.comm_range = float(comm_range)
        self.seed = seed
        self.ai = ai
        self.auto_rate = auto_rate
        self.default_ttl = default_ttl
        self.max_transfers = max_transfers
        self.gen_until = gen_until
        self.ai_margin = ai_margin
        # Separate RNGs: mobility and message schedule are identical for every
        # protocol with the same seed, so comparisons are fair.
        self.move_rng = random.Random(seed)
        self.msg_rng = random.Random(seed + 1)
        self.route_rng = random.Random(seed + 2)

        self.t = 0
        self.nodes: List[Node] = []
        self.messages: Dict[str, Message] = {}
        self.events: List[dict] = []
        self.history: List[dict] = []
        self.contacts: List[Tuple[int, int]] = []
        self.active_pairs: set = set()
        self.total_encounters = 0
        self.transmissions = 0
        self.duplicates_blocked = 0
        self.last_ai: Optional[dict] = None
        self.msg_counter = 1
        self.moved: set = set()
        self._build_nodes(n_nodes, buffer_size)

    # ------------------------------------------------------------------ setup
    def _build_nodes(self, n_nodes, buffer_size):
        rng = self.move_rng
        a = self.area
        hospitals = [(0.85 * a, 0.85 * a), (0.15 * a, 0.20 * a)]
        n_vehicles = max(2, round(n_nodes * 0.25))
        n_users = n_nodes - n_vehicles - len(hospitals)
        nid = 0
        for i, (hx, hy) in enumerate(hospitals):
            self.nodes.append(Node(nid, "hospital", f"Hospital-{i+1}", f"H{i+1}", hx, hy,
                                   0.0, 30, 100.0))
            nid += 1
        for i in range(n_vehicles):
            n = Node(nid, "vehicle", f"Rescue-{i+1}", f"R{i+1}", rng.uniform(5, a - 5),
                     rng.uniform(5, a - 5), rng.uniform(3.5, 6.0), buffer_size * 2,
                     rng.uniform(80, 100))
            self._new_target(n)
            self.nodes.append(n)
            nid += 1
        for i in range(n_users):
            n = Node(nid, "user", f"User-{i+1}", f"U{i+1}", rng.uniform(5, a - 5),
                     rng.uniform(5, a - 5), rng.uniform(0.8, 1.6), buffer_size,
                     rng.uniform(60, 100))
            self._new_target(n)
            self.nodes.append(n)
            nid += 1

    def _new_target(self, n: Node):
        rng, a = self.move_rng, self.area
        if n.kind == "vehicle":
            if rng.random() < 0.35:   # vehicles regularly head to a hospital
                h = rng.choice([x for x in self.nodes if x.kind == "hospital"])
                n.target = (min(a - 2, max(2, h.x + rng.uniform(-6, 6))),
                            min(a - 2, max(2, h.y + rng.uniform(-6, 6))))
            else:
                n.target = (rng.uniform(5, a - 5), rng.uniform(5, a - 5))
        else:  # users wander locally
            n.target = (min(a - 2, max(2, n.x + rng.uniform(-30, 30))),
                        min(a - 2, max(2, n.y + rng.uniform(-30, 30))))

    # --------------------------------------------------------------- messages
    def node_by_name(self, name: str) -> Node:
        return next(n for n in self.nodes if n.name == name)

    def _log(self, etype, msg_id, frm=None, to=None, prob=None, detail=""):
        fx = fy = tx = ty = None
        if frm is not None:
            fx, fy = self.nodes[frm].x, self.nodes[frm].y
        if to is not None:
            tx, ty = self.nodes[to].x, self.nodes[to].y
        self.events.append({
            "step": self.t, "type": etype, "msg_id": msg_id,
            "from": self.nodes[frm].name if frm is not None else "",
            "to": self.nodes[to].name if to is not None else "",
            "prob": prob, "detail": detail, "fx": fx, "fy": fy, "tx": tx, "ty": ty,
        })

    def create_message(self, source: int, destination: int, priority="HIGH",
                       ttl: Optional[int] = None, text="") -> Message:
        ttl = int(ttl if ttl is not None else self.default_ttl)
        msg = Message(f"MSG-{self.msg_counter:03d}", source, destination, priority, ttl,
                      self.t, text or f"Emergency from {self.nodes[source].name}")
        self.msg_counter += 1
        self.messages[msg.msg_id] = msg
        src = self.nodes[source]
        msg.path = [source]
        msg.path_points = [(src.x, src.y)]
        self._log("CREATED", msg.msg_id, source, destination,
                  detail=f"{priority} priority, TTL {ttl}")
        if self._can_accept(src, msg, count_dup=False):
            self._store(src, msg)
        else:
            msg.status = "DROPPED"
            self._log("DROPPED", msg.msg_id, source, detail="source buffer full")
        return msg

    def _auto_generate(self):
        if self.auto_rate <= 0 or (self.gen_until is not None and self.t > self.gen_until):
            return
        if self.msg_rng.random() >= self.auto_rate:
            return
        users = [n.id for n in self.nodes if n.kind == "user"]
        hospitals = [n.id for n in self.nodes if n.kind == "hospital"]
        prio = self.msg_rng.choices(PRIORITIES, weights=[0.3, 0.4, 0.3])[0]
        src = self.msg_rng.choice(users)
        dst = self.msg_rng.choice(hospitals)
        self.create_message(src, dst, prio, int(self.default_ttl * TTL_FACTOR[prio]))

    # --------------------------------------------------------------- buffering
    def _can_accept(self, node: Node, msg: Message, count_dup=True) -> bool:
        if not node.alive or node.battery <= 3:
            return False
        if node.kind == "hospital" and msg.destination != node.id:
            return False  # hospitals are gateways, not relays
        if msg.msg_id in node.seen:
            if count_dup:
                self.duplicates_blocked += 1
            return False
        if len(node.buffer) < node.capacity:
            return True
        return min(m.rank for m in node.buffer.values()) < msg.rank

    def _store(self, node: Node, msg: Message):
        """STORE: put bundle in node buffer, evicting the least important if full."""
        if len(node.buffer) >= node.capacity:
            victim = min(node.buffer.values(),
                         key=lambda m: (m.rank, m.created_step + m.ttl))
            del node.buffer[victim.msg_id]
            victim.status = "DROPPED"
            victim.holder = None
            self._log("DROPPED", victim.msg_id, node.id, detail="buffer overflow (lower priority evicted)")
        node.buffer[msg.msg_id] = msg
        node.seen.add(msg.msg_id)
        node.relay_carried += 1
        msg.holder = node.id

    # ---------------------------------------------------------------- features
    def _features(self, node: Node, msg: Message, extra_hop: int) -> List[float]:
        dest = self.nodes[msg.destination]
        dist = math.hypot(node.x - dest.x, node.y - dest.y) / self.diag
        enc_dest = min(1.0, node.enc_with.get(dest.id, 0) / 6.0)
        enc_all = min(1.0, node.encounters / 60.0)
        return [
            dist,
            max(0.0, node.battery) / 100.0,
            1 - len(node.buffer) / node.capacity,
            0.8 * enc_dest + 0.2 * enc_all,
            node.delivery_rate,
            msg.hops + extra_hop,
        ]

    # --------------------------------------------------------------- transfers
    def _deliver(self, msg: Message, sender: Node, dest: Node):
        del sender.buffer[msg.msg_id]
        msg.status = "DELIVERED"
        msg.delivered_step = self.t
        msg.hops += 1
        msg.path.append(dest.id)
        msg.path_points.append((dest.x, dest.y))
        msg.holder = None
        dest.seen.add(msg.msg_id)
        dest.delivered_to_me.append(msg.msg_id)
        for nid in set(msg.path[:-1]):
            self.nodes[nid].relay_success += 1
            self.nodes[nid].delivery_history.append(msg.msg_id)
        self._radio_cost(sender, dest)
        self._log("DELIVERED", msg.msg_id, sender.id, dest.id, prob=1.0,
                  detail=f"{msg.hops} hops, delay {self.t - msg.created_step} min")

    def _forward(self, msg: Message, sender: Node, receiver: Node, prob: Optional[float]):
        del sender.buffer[msg.msg_id]
        self._store(receiver, msg)
        msg.hops += 1
        msg.path.append(receiver.id)
        msg.path_points.append((receiver.x, receiver.y))
        if prob is not None:
            msg.ai_prob = prob
            self.last_ai = {"step": self.t, "msg_id": msg.msg_id, "from": sender.id,
                            "to": receiver.id, "prob": prob}
        self._radio_cost(sender, receiver)
        self._log("FORWARD", msg.msg_id, sender.id, receiver.id, prob=prob,
                  detail=f"hop {msg.hops}")

    def _radio_cost(self, sender: Node, receiver: Node):
        self.transmissions += 1
        sender.tx += 1
        receiver.rx += 1
        if sender.kind != "hospital":
            sender.battery = max(0.0, sender.battery - 0.20)
        if receiver.kind != "hospital":
            receiver.battery = max(0.0, receiver.battery - 0.10)

    def _process_sender(self, sender: Node, neighbors: List[Node]):
        if not sender.buffer or not sender.alive:
            return
        budget = self.max_transfers
        by_id = {n.id: n for n in neighbors}
        msgs = sorted(sender.buffer.values(), key=lambda m: (-m.rank, m.created_step))
        remaining = []
        # 1) Direct hand-over to the destination whenever it is in range
        for m in msgs:
            dest = by_id.get(m.destination)
            if dest is not None and dest.alive and budget > 0 and m.msg_id not in self.moved:
                self._deliver(m, sender, dest)
                self.moved.add(m.msg_id)
                budget -= 1
            else:
                remaining.append(m)
        if self.protocol == "Direct Delivery" or budget <= 0:
            return
        remaining = [m for m in remaining if m.msg_id not in self.moved]
        relays = [n for n in neighbors if n.kind != "hospital" and n.alive and n.battery > 3]
        if not remaining or not relays:
            return

        scores = None
        if self.protocol == "AI Routing":
            rows = []
            for m in remaining:
                rows.append(self._features(sender, m, 0))
                for r in relays:
                    rows.append(self._features(r, m, 1))
            probs = self.ai.predict_proba(np.array(rows))
            scores, k = {}, 0
            width = 1 + len(relays)
            for m in remaining:
                scores[m.msg_id] = probs[k:k + width]
                m.ai_prob = float(probs[k])
                k += width

        for m in remaining:
            if budget <= 0:
                break
            idx = [i for i, r in enumerate(relays) if self._can_accept(r, m)]
            if not idx:
                continue
            if self.protocol == "Random Routing":
                if self.route_rng.random() < 0.5:
                    self._forward(m, sender, relays[self.route_rng.choice(idx)], None)
                    self.moved.add(m.msg_id)
                    budget -= 1
            else:  # AI routing: pick the best-scoring neighbour
                sc = scores[m.msg_id]
                best = max(idx, key=lambda i: sc[i + 1])
                if sc[best + 1] > sc[0] + self.ai_margin:
                    self._forward(m, sender, relays[best], float(sc[best + 1]))
                    self.moved.add(m.msg_id)
                    budget -= 1

    # -------------------------------------------------------------------- step
    def _move(self):
        for n in self.nodes:
            if n.kind == "hospital":
                continue
            dx, dy = n.target[0] - n.x, n.target[1] - n.y
            d = math.hypot(dx, dy)
            if d <= n.speed:
                n.x, n.y = n.target
                self._new_target(n)
            else:
                n.x += dx / d * n.speed
                n.y += dy / d * n.speed

    def _expire(self):
        for m in self.messages.values():
            if m.status == "PENDING" and self.t - m.created_step >= m.ttl:
                m.status = "EXPIRED"
                if m.holder is not None:
                    self.nodes[m.holder].buffer.pop(m.msg_id, None)
                self._log("EXPIRED", m.msg_id, m.holder, detail="TTL reached")
                m.holder = None

    def _detect_contacts(self):
        P = np.array([[n.x, n.y] for n in self.nodes])
        alive = np.array([n.alive for n in self.nodes])
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        ii, jj = np.where(np.triu(D <= self.comm_range, k=1) & alive[:, None] & alive[None, :])
        self.contacts = list(zip(ii.tolist(), jj.tolist()))
        pairs = set(self.contacts)
        for i, j in pairs - self.active_pairs:   # newly started encounters
            a, b = self.nodes[i], self.nodes[j]
            a.encounters += 1
            b.encounters += 1
            a.enc_with[j] = a.enc_with.get(j, 0) + 1
            b.enc_with[i] = b.enc_with.get(i, 0) + 1
            self.total_encounters += 1
        self.active_pairs = pairs

    def step(self):
        self.t += 1
        self.moved = set()
        self._auto_generate()
        self._move()
        self._expire()
        self._detect_contacts()
        nbrs: Dict[int, List[Node]] = {n.id: [] for n in self.nodes}
        for i, j in self.contacts:
            nbrs[i].append(self.nodes[j])
            nbrs[j].append(self.nodes[i])
        for n in self.nodes:
            if n.kind != "hospital":
                n.battery = max(0.0, n.battery - (0.015 if n.kind == "vehicle" else 0.03))
        for n in self.nodes:
            self._process_sender(n, nbrs[n.id])
        m = self.metrics()
        self.history.append({"step": self.t, "delivered": m["delivered"], "pending": m["pending"],
                             "expired": m["expired"], "dropped": m["dropped"],
                             "delivery_ratio": m["delivery_ratio"]})

    # ----------------------------------------------------------------- reports
    def metrics(self) -> dict:
        msgs = list(self.messages.values())
        total = len(msgs)
        deliv = [m for m in msgs if m.status == "DELIVERED"]
        pending = [m for m in msgs if m.status == "PENDING"]
        probs = [m.ai_prob for m in pending if m.ai_prob is not None]
        return {
            "total": total,
            "delivered": len(deliv),
            "pending": len(pending),
            "expired": sum(m.status == "EXPIRED" for m in msgs),
            "dropped": sum(m.status == "DROPPED" for m in msgs),
            "delivery_ratio": len(deliv) / total if total else 0.0,
            "avg_delay": float(np.mean([m.delivered_step - m.created_step for m in deliv])) if deliv else float("nan"),
            "avg_hops": float(np.mean([m.hops for m in deliv])) if deliv else float("nan"),
            "transmissions": self.transmissions,
            "avg_ai_prob": float(np.mean(probs)) if probs else float("nan"),
        }

    def priority_stats(self) -> pd.DataFrame:
        rows = []
        for p in PRIORITIES:
            ms = [m for m in self.messages.values() if m.priority == p]
            d = [m for m in ms if m.status == "DELIVERED"]
            rows.append({
                "Priority": p, "Messages": len(ms), "Delivered": len(d),
                "Delivery ratio (%)": round(100 * len(d) / len(ms), 1) if ms else 0.0,
                "Avg delay (min)": round(float(np.mean([m.delivered_step - m.created_step for m in d])), 1) if d else None,
            })
        return pd.DataFrame(rows)

    def route_names(self, m: Message) -> str:
        return " → ".join(self.nodes[i].name for i in m.path)

    def messages_df(self) -> pd.DataFrame:
        rows = []
        for m in self.messages.values():
            rows.append({
                "ID": m.msg_id, "Source": self.nodes[m.source].name,
                "Destination": self.nodes[m.destination].name, "Priority": m.priority,
                "TTL": m.ttl, "Created": m.created_step,
                "Age": (m.delivered_step if m.delivered_step is not None else self.t) - m.created_step,
                "Status": m.status,
                "Carrier": self.nodes[m.holder].name if m.holder is not None else "-",
                "Hops": m.hops,
                "AI prob": round(m.ai_prob, 3) if m.ai_prob is not None else None,
                "Route": self.route_names(m),
            })
        return pd.DataFrame(rows)

    def nodes_df(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Node": n.name, "Type": n.kind, "X": round(n.x, 1), "Y": round(n.y, 1),
            "Battery %": round(n.battery, 1), "Buffer": f"{len(n.buffer)}/{n.capacity}",
            "Encounters": n.encounters, "Carried": n.relay_carried,
            "Helped deliver": n.relay_success, "Delivery rate": round(n.delivery_rate, 2),
            "Received (as dest)": len(n.delivered_to_me), "Tx": n.tx, "Rx": n.rx,
        } for n in self.nodes])

    def events_df(self) -> pd.DataFrame:
        cols = ["step", "type", "msg_id", "from", "to", "prob", "detail"]
        if not self.events:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(self.events)[cols]

    def messages_records(self) -> List[dict]:
        return [{
            "msg_id": m.msg_id, "source": self.nodes[m.source].name,
            "destination": self.nodes[m.destination].name, "priority": m.priority,
            "ttl": m.ttl, "created_step": m.created_step, "status": m.status,
            "delivered_step": m.delivered_step, "hops": m.hops, "path": self.route_names(m),
        } for m in self.messages.values()]

    def events_records(self) -> List[dict]:
        return [{k: e[k] for k in ("step", "type", "msg_id", "from", "to", "prob", "detail")}
                for e in self.events]


# ------------------------------------------------------------------ comparison
def run_headless(protocol, steps, seed, ai, **kw) -> Simulation:
    sim = Simulation(protocol=protocol, seed=seed, ai=ai, gen_until=int(steps * 0.7), **kw)
    for _ in range(steps):
        sim.step()
    return sim


def compare_protocols(ai, steps=400, seeds=(1, 2, 3), progress=None, **kw):
    """Run every protocol on identical mobility + message schedules.

    Returns (summary_df, per_run_df, series_df).
    """
    per_run, series = [], []
    total_runs = len(PROTOCOLS) * len(seeds)
    done = 0
    for proto in PROTOCOLS:
        for sd in seeds:
            sim = run_headless(proto, steps, sd, ai, **kw)
            m = sim.metrics()
            per_run.append({
                "Protocol": proto, "Seed": sd, "Total": m["total"], "Delivered": m["delivered"],
                "Expired": m["expired"], "Dropped": m["dropped"],
                "Delivery Ratio (%)": 100 * m["delivery_ratio"],
                "Avg Delay (min)": m["avg_delay"], "Avg Hops": m["avg_hops"],
                "Transmissions": m["transmissions"],
                "Tx per Delivered": m["transmissions"] / m["delivered"] if m["delivered"] else np.nan,
            })
            for h in sim.history[::10]:
                series.append({"Protocol": proto, "Seed": sd, "Step": h["step"],
                               "Delivery Ratio (%)": 100 * h["delivery_ratio"]})
            done += 1
            if progress:
                progress(done / total_runs)
    runs = pd.DataFrame(per_run)
    summary = runs.groupby("Protocol", sort=False).agg(
        **{"Delivery Ratio (%)": ("Delivery Ratio (%)", "mean"),
           "Ratio Std": ("Delivery Ratio (%)", "std"),
           "Avg Delay (min)": ("Avg Delay (min)", "mean"),
           "Avg Hops": ("Avg Hops", "mean"),
           "Delivered": ("Delivered", "mean"),
           "Total": ("Total", "mean"),
           "Transmissions": ("Transmissions", "mean"),
           "Tx per Delivered": ("Tx per Delivered", "mean")}
    ).reset_index().fillna(0.0)
    ser = pd.DataFrame(series).groupby(["Protocol", "Step"], sort=False)["Delivery Ratio (%)"].mean().reset_index()
    return summary, runs, ser
