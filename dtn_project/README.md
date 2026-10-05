# AI-Optimized Delay Tolerant Network for Disaster Communication

A working **Python + Streamlit simulation** of emergency messaging when no continuous Internet exists.
Messages follow **STORE → CARRY → FORWARD** between moving nodes
(`User → Node → Rescue Vehicle → Node → Hospital`). A **Random Forest** picks the best relay.
This is a simulation/prototype, not a real Bluetooth/Wi-Fi deployment.

## Installation
```bash
cd dtn_project
python -m venv venv                # optional
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Opens at http://localhost:8501. Python 3.9+ recommended. The SQLite file `dtn_results.db` is created automatically.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard (Plotly map, KPIs, tabs, comparison charts) |
| `dtn.py` | Simulation engine: mobility, range, encounters, buffers, TTL, duplicates, priority, 3 routing protocols |
| `ai_router.py` | Synthetic dataset generator + Random Forest delivery-probability model |
| `database.py` | SQLite storage for runs, messages and events |

## Architecture
```
 Streamlit UI (app.py)
   | controls / charts
   v
 Simulation (dtn.py) --- scores candidates ---> AIRouter (ai_router.py, Random Forest)
   | save runs/messages/events
   v
 SQLite (database.py -> dtn_results.db)
```
* Area 100x100 units (about 10 km). 1 step = 1 simulated minute.
* Node types: 2 static **Hospitals** (gateways/destinations), ~25% fast **Rescue vehicles**
  (patrol and visit hospitals), the rest slow **User** nodes wandering locally.
* Mobility RNG and message schedule are seeded separately from routing, so every protocol sees
  **identical movement and traffic** -> fair comparison.

## How the DTN works
1. **Store** - a message (ID, source, destination, priority, TTL) is placed in the source buffer.
2. **Carry** - nodes keep moving while holding the message (battery drains with movement/radio use).
3. **Forward** - an *encounter* happens when two nodes are within communication range. The carrier
   may hand the message over (single custody copy) or deliver it if the destination is in range.
* **TTL**: message expires when age >= TTL (removed from buffer, status EXPIRED).
* **Duplicate detection**: each node keeps a `seen` list and refuses a bundle it already held (stops loops).
* **Priority**: HIGH is forwarded first, gets longer default TTL, and low-priority bundles are evicted first
  when a buffer is full (status DROPPED). Per-contact bandwidth is limited to 3 transfers.
* Hospitals only accept messages addressed to them (they are gateways, not relays).

## How AI routing works
For each pending message the carrier builds a feature row for itself and for every neighbour in range:

| Feature | Meaning |
|---|---|
| distance | candidate's distance to destination (normalised) |
| battery | candidate battery 0-1 |
| buffer availability | free buffer fraction |
| encounter frequency | how often candidate has met the destination (80%) / anyone (20%) |
| previous delivery rate | smoothed (successes+1)/(carried+2) |
| hop count | hops the message would have after the transfer |

The Random Forest (40 trees, depth 8) returns P(delivery). The neighbour with the highest P is chosen and the
message is forwarded only if `P(best) > P(carrier) + 0.05` (margin avoids ping-pong). Otherwise it keeps carrying.

**Training data is SIMULATED.** No real disaster trace exists, so `ai_router.py` generates 4,000 synthetic samples from
a noisy logistic rule and trains on them (hold-out accuracy/AUC shown in the dashboard). The dashboard states this clearly.

## Protocols compared
* **Direct Delivery** - source keeps the message until it meets the destination.
* **Random Routing** - forwards to a random neighbour with probability 0.5 per contact.
* **AI Routing** - Random Forest score as above.
Typical result (400 min, 3 seeds, defaults): Direct about 40% delivery, Random about 60%, AI about 90%, with AI also using far
fewer transmissions than Random.

## How to demonstrate in the viva
1. Run `streamlit run app.py`; explain the banner: no Internet, store-carry-forward.
2. Sidebar: show 15 nodes, range, TTL. Point out colours: blue users, orange rescue vehicles, green hospitals.
3. **Create message** tab -> click *Inject demo scenario*. Message is stored at a user (purple ring = carrying).
4. Click **Animate**. Watch the purple dashed path, red arrows (AI forwards), the gold ring *AI pick p=...*, and green arrow on delivery.
5. Show **Messages** tab (route `User-1 → Rescue-2 → Hospital-1`, hops, delay) and **Nodes** tab (battery, buffer, encounters, delivery history).
6. Create a LOW and a HIGH message to show priority; set TTL to 20 to show expiry; use a tiny buffer to show DROPPED.
7. **AI router** tab: feature importance, accuracy/AUC, the probability gauge; say clearly the data is synthetic.
8. Switch the sidebar protocol live (Direct/Random/AI) and reset to show different behaviour.
9. **Protocol comparison** tab -> *Run comparison*; discuss delivery ratio, delay, overhead and over-time chart.
10. **Logs & database** tab: save a run and show the SQLite tables.

Likely questions: *Why not Epidemic routing?* (high overhead - we show Tx per delivered). *Why avg delay lower for Direct sometimes?*
(only delivered messages are averaged - read with delivery ratio). *Is the AI really learning?* (learns from simulated data - limitation).

## Limitations
* Simulated radio: perfect link inside range, no interference, fixed transfer budget, no bundle sizes.
* Mobility is simple random-waypoint; no roads, obstacles or real GPS traces.
* Single-copy forwarding only (no replication like Spray-and-Wait/Epidemic).
* Model trained on synthetic data; no online learning.
* Battery model is simplified; no security/authentication.

## Future scope
* Train on logged simulation outcomes or real traces (e.g., ONE simulator, CRAWDAD).
* Multi-copy / Spray-and-Wait hybrids, Gradient Boosting or reinforcement learning, PRoPHET baseline.
* Realistic mobility (road networks, SUMO), radio models (BLE/Wi-Fi Direct), message sizes and bandwidth.
* Real Android/Raspberry Pi prototype, encryption, GPS-based drone relays.
