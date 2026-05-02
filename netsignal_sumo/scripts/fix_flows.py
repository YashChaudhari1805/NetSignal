#!/usr/bin/env python3
"""
fix_flows.py  (v3 - uses real edge IDs from net.xml)
Run once after generate_network.py:
    python netsignal_sumo/scripts/fix_flows.py
"""
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
NET_FILE    = PROJECT_DIR / "network" / "grid6x6.net.xml"
FLOW_FILE   = PROJECT_DIR / "flows" / "flow.xml"

def main():
    tree = ET.parse(NET_FILE)
    root = tree.getroot()

    # Get all non-internal edges
    edges = [e for e in root.findall(".//edge") if not e.get("id","").startswith(":")]
    edge_by_nodes = {(e.get("from"), e.get("to")): e.get("id") for e in edges}
    print(f"Loaded {len(edges)} edges")
    print("Sample edges:")
    for (f,t),eid in list(edge_by_nodes.items())[:6]:
        print(f"  {f} -> {t} : {eid}")

    # Get TLS junctions sorted into 6x6 grid by x,y
    junctions = [(j.get("id"), float(j.get("x",0)), float(j.get("y",0)))
                 for j in root.findall(".//junction") if j.get("type")=="traffic_light"]

    xs = sorted(set(round(x) for _,x,_ in junctions))
    ys = sorted(set(round(y) for _,_,y in junctions))
    # Cluster into 6 columns and 6 rows
    from itertools import groupby
    def cluster(vals, n):
        vals = sorted(vals)
        step = (vals[-1]-vals[0]) / (n-1) if n > 1 else 1
        result = []
        for v in vals:
            idx = round((v - vals[0]) / step) if step else 0
            result.append(min(idx, n-1))
        return dict(zip(vals, result))
    xmap = cluster(xs, 6)
    ymap = cluster(ys, 6)

    grid = {}
    for jid, x, y in junctions:
        col = xmap[round(x)]
        row = ymap[round(y)]
        grid[(col, row)] = jid

    GRID = 6
    print("\nGrid (row 0=south):")
    for r in range(GRID):
        print("  "+"  ".join(f"{grid.get((c,r),'?'):>4}" for c in range(GRID)))

    def get_edge(c1,r1,c2,r2):
        n1 = grid.get((c1,r1))
        n2 = grid.get((c2,r2))
        if not n1 or not n2: return None
        return edge_by_nodes.get((n1,n2))

    # Build straight-line routes across the full grid
    routes = {}
    # N->S (row 0 to row 5) per column
    for c in range(GRID):
        segs = [get_edge(c,r,c,r+1) for r in range(GRID-1)]
        if all(segs): routes[f"r_NS_{c}"] = " ".join(segs)
        else:
            # Try S->N direction  
            segs2 = [get_edge(c,r+1,c,r) for r in range(GRID-2,-1,-1)]
            if all(segs2): routes[f"r_NS_{c}"] = " ".join(segs2)
            else: print(f"  WARNING: col {c} N-S missing: {segs}")

    # S->N per column
    for c in range(GRID):
        segs = [get_edge(c,r,c,r-1) for r in range(GRID-1,0,-1)]
        if all(segs): routes[f"r_SN_{c}"] = " ".join(segs)

    # E->W (col 0 to col 5) per row
    for r in range(GRID):
        segs = [get_edge(c,r,c+1,r) for c in range(GRID-1)]
        if all(segs): routes[f"r_EW_{r}"] = " ".join(segs)
        else:
            segs2 = [get_edge(c+1,r,c,r) for c in range(GRID-2,-1,-1)]
            if all(segs2): routes[f"r_EW_{r}"] = " ".join(segs2)
            else: print(f"  WARNING: row {r} E-W missing: {segs}")

    # W->E per row
    for r in range(GRID):
        segs = [get_edge(c,r,c-1,r) for c in range(GRID-1,0,-1)]
        if all(segs): routes[f"r_WE_{r}"] = " ".join(segs)

    # EV route from (0,0) to (3,3)
    ev_segs = [get_edge(0,0,1,0), get_edge(1,0,2,0), get_edge(2,0,2,1),
               get_edge(2,1,2,2), get_edge(2,2,3,2), get_edge(3,2,3,3)]
    if all(ev_segs):
        routes["r_EV"] = " ".join(ev_segs)
    else:
        # fallback: find any path col0->col3
        print(f"  EV route segs: {ev_segs}")

    print(f"\nBuilt {len(routes)} routes:")
    for rid in list(routes)[:4]:
        print(f"  {rid}: {routes[rid][:60]}...")

    # Write flow.xml
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
        '',
        '    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="4.5" minGap="2.5" maxSpeed="13.89" speedFactor="normc(1,0.1,0.8,1.2)" color="0.3,0.6,1.0"/>',
        '    <vType id="truck" accel="1.3" decel="3.5" sigma="0.3" length="10.0" minGap="4.0" maxSpeed="10.0" color="0.7,0.4,0.1"/>',
        '    <vType id="emergency" accel="4.0" decel="6.0" sigma="0.1" length="6.5" minGap="1.5" maxSpeed="20.0" color="1.0,0.0,0.0" guiShape="emergency"/>',
        '',
    ]

    for rid, edges_str in routes.items():
        lines.append(f'    <route id="{rid}" edges="{edges_str}"/>')
    lines.append('')

    fid = 0
    for c in range(GRID):
        if f"r_NS_{c}" in routes:
            lines.append(f'    <flow id="f{fid}" type="car" route="r_NS_{c}" begin="0"    end="1800" period="6" departLane="best" departSpeed="avg"/>'); fid+=1
            lines.append(f'    <flow id="f{fid}" type="car" route="r_NS_{c}" begin="1800" end="3600" period="9" departLane="best" departSpeed="avg"/>'); fid+=1
        if f"r_SN_{c}" in routes:
            lines.append(f'    <flow id="f{fid}" type="car" route="r_SN_{c}" begin="0"    end="1800" period="6" departLane="best" departSpeed="avg"/>'); fid+=1
            lines.append(f'    <flow id="f{fid}" type="car" route="r_SN_{c}" begin="1800" end="3600" period="9" departLane="best" departSpeed="avg"/>'); fid+=1

    for r in range(GRID):
        if f"r_EW_{r}" in routes:
            lines.append(f'    <flow id="f{fid}" type="car" route="r_EW_{r}" begin="0"    end="1800" period="6" departLane="best" departSpeed="avg"/>'); fid+=1
            lines.append(f'    <flow id="f{fid}" type="car" route="r_EW_{r}" begin="1800" end="3600" period="9" departLane="best" departSpeed="avg"/>'); fid+=1
        if f"r_WE_{r}" in routes:
            lines.append(f'    <flow id="f{fid}" type="car" route="r_WE_{r}" begin="0"    end="1800" period="6" departLane="best" departSpeed="avg"/>'); fid+=1
            lines.append(f'    <flow id="f{fid}" type="car" route="r_WE_{r}" begin="1800" end="3600" period="9" departLane="best" departSpeed="avg"/>'); fid+=1

    lines.append(f'    <flow id="f{fid}" type="truck" route="r_NS_2" begin="0" end="3600" period="30" departLane="0" departSpeed="avg"/>'); fid+=1
    ev_route = "r_EV" if "r_EV" in routes else list(routes.keys())[0]
    lines.append(f'    <vehicle id="EV_001" type="emergency" route="{ev_route}" depart="900" departLane="0" departSpeed="max" color="1,0,0"/>')
    lines += ['', '</routes>']

    FLOW_FILE.write_text("\n".join(lines), encoding="utf-8")

    # Save node map
    cfg = PROJECT_DIR / "config" / "grid_nodes.txt"
    with open(cfg, "w") as f:
        f.write(f"HOSPITAL_NODE={grid.get((3,3),'D3')}\n")
        f.write(f"EV_START_NODE={grid.get((0,0),'A0')}\n")
        for r in range(GRID):
            for c in range(GRID):
                f.write(f"NODE_{c}_{r}={grid.get((c,r),'')}\n")

    print(f"\nWrote {FLOW_FILE}")
    print(f"Hospital: {grid.get((3,3))}, EV start: {grid.get((0,0))}")
    print("Run: python netsignal_sumo/scripts/traciloop.py --gui")

if __name__ == "__main__":
    main()