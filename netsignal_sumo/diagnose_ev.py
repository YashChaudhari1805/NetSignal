#!/usr/bin/env python3
"""
diagnose_ev.py
==============
Run this from your project root to diagnose why EV_001 is not appearing.
It checks the actual edge IDs in your network and fixes the flow.xml.

Usage:
    python diagnose_ev.py
"""

import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
NET_FILE    = PROJECT_DIR / "network" / "grid6x6.net.xml"
FLOW_FILE   = PROJECT_DIR / "flows" / "flow.xml"

# ---------------------------------------------------------------------------
# Step 1: Parse junctions and find real IDs for grid positions
# ---------------------------------------------------------------------------
print("=" * 60)
print("STEP 1: Reading junctions from network")
print("=" * 60)

root = ET.parse(NET_FILE).getroot()

tl_junctions = [
    (j.get("id"), float(j.get("x")), float(j.get("y")))
    for j in root.findall(".//junction")
    if j.get("type") == "traffic_light"
]

# Sort and cluster into 6x6 grid
xs = sorted(set(round(x) for _, x, _ in tl_junctions))
ys = sorted(set(round(y) for _, _, y in tl_junctions))

def cluster(values, n):
    step = (values[-1] - values[0]) / (n - 1)
    return {v: min(round((v - values[0]) / step), n - 1) for v in values}

col_map = cluster(xs, 6)
row_map = cluster(ys, 6)

grid = {}
for jid, x, y in tl_junctions:
    col = col_map[round(x)]
    row = row_map[round(y)]
    grid[(col, row)] = jid

print("\nGrid layout (col, row) -> junction_id:")
for row in range(6):
    for col in range(6):
        print(f"  ({col},{row}) = {grid.get((col,row), 'MISSING')}")

# Key nodes for EV route
print("\nEV route nodes:")
ev_nodes = [(0,0),(1,0),(2,0),(2,1),(2,2),(3,2),(3,3)]
for pos in ev_nodes:
    print(f"  {pos} -> {grid.get(pos, 'MISSING')}")

# ---------------------------------------------------------------------------
# Step 2: Find real edge IDs between consecutive EV nodes
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Finding real edge IDs for EV route")
print("=" * 60)

edges_by_nodes = {}
for edge in root.findall(".//edge"):
    eid = edge.get("id", "")
    if not eid.startswith(":"):
        edges_by_nodes[(edge.get("from"), edge.get("to"))] = eid

ev_edges = []
ok = True
for i in range(len(ev_nodes) - 1):
    n1 = grid.get(ev_nodes[i])
    n2 = grid.get(ev_nodes[i+1])
    eid = edges_by_nodes.get((n1, n2))
    status = "OK" if eid else "MISSING"
    print(f"  {ev_nodes[i]}->{ev_nodes[i+1]}  ({n1}->{n2})  edge={eid}  [{status}]")
    if eid:
        ev_edges.append(eid)
    else:
        ok = False

# ---------------------------------------------------------------------------
# Step 3: Check current flow.xml for EV_001
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Checking current flow.xml")
print("=" * 60)

flow_root = ET.parse(FLOW_FILE).getroot()
ev_vehicle = flow_root.find(".//vehicle[@id='EV_001']")
if ev_vehicle is not None:
    print(f"  EV_001 found: depart={ev_vehicle.get('depart')} route={ev_vehicle.get('route')}")
    route_id = ev_vehicle.get("route")
    route_el = flow_root.find(f".//route[@id='{route_id}']")
    if route_el is not None:
        print(f"  Route edges: {route_el.get('edges')}")
    else:
        print(f"  Route '{route_id}' NOT FOUND in flow.xml")
else:
    print("  EV_001 NOT FOUND in flow.xml!")

# ---------------------------------------------------------------------------
# Step 4: Rewrite flow.xml with corrected EV route if needed
# ---------------------------------------------------------------------------
if ok and ev_edges:
    corrected_route = " ".join(ev_edges)
    print("\n" + "=" * 60)
    print("STEP 4: Writing corrected flow.xml")
    print("=" * 60)
    print(f"  Correct EV edge string: {corrected_route}")

    # Read and patch flow.xml text directly
    text = FLOW_FILE.read_text(encoding="utf-8")

    # Fix the r_EV route
    import re
    text = re.sub(
        r'<route id="r_EV"[^/]*/?>',
        f'<route id="r_EV" edges="{corrected_route}"/>',
        text
    )

    # Make sure EV_001 uses r_EV and has correct depart
    text = re.sub(
        r'<vehicle id="EV_001"[^/]*/?>',
        f'<vehicle id="EV_001" type="emergency" route="r_EV" depart="900" departLane="0" departSpeed="max" color="1,0,0"/>',
        text
    )

    FLOW_FILE.write_text(text, encoding="utf-8")
    print("  flow.xml patched successfully.")
    print("\n  Now run:  python scripts/traciloop.py --gui")
    print("  Wait for t=900s — the red ambulance will appear at node", grid.get((0,0)))

elif not ok:
    print("\n" + "=" * 60)
    print("STEP 4: Cannot auto-fix — some EV route edges are missing")
    print("=" * 60)
    print("  The edge IDs between those nodes don't exist in the network.")
    print("  You may need to re-run:  python scripts/generate_network.py")
    print("  Then:                    python scripts/fix_flows.py")
