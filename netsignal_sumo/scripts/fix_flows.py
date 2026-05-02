#!/usr/bin/env python3
"""
fix_flows.py  (patched — routes sorted by departure time)
"""

import xml.etree.ElementTree as ET
from pathlib import Path


_PROJECT_DIR = Path(__file__).parent.parent.resolve()
_NET_FILE    = _PROJECT_DIR / "network" / "grid6x6.net.xml"
_FLOW_FILE   = _PROJECT_DIR / "flows" / "flow.xml"
_NODE_FILE   = _PROJECT_DIR / "config" / "grid_nodes.txt"

_GRID_SIZE = 6


def _parse_edges(root):
    return {
        (edge.get("from"), edge.get("to")): edge.get("id")
        for edge in root.findall(".//edge")
        if not edge.get("id", "").startswith(":")
    }


def _parse_grid(root):
    junctions = [
        (j.get("id"), float(j.get("x", 0)), float(j.get("y", 0)))
        for j in root.findall(".//junction")
        if j.get("type") == "traffic_light"
    ]

    xs = sorted({round(x) for _, x, _ in junctions})
    ys = sorted({round(y) for _, _, y in junctions})

    def _cluster(values, n):
        step = (values[-1] - values[0]) / (n - 1) if n > 1 else 1
        return {
            v: min(round((v - values[0]) / step) if step else 0, n - 1)
            for v in values
        }

    col_map = _cluster(xs, _GRID_SIZE)
    row_map = _cluster(ys, _GRID_SIZE)

    grid = {}
    for jid, x, y in junctions:
        col = col_map[round(x)]
        row = row_map[round(y)]
        grid[(col, row)] = jid

    return grid


def _build_routes(grid, edge_by_nodes):
    G = _GRID_SIZE

    def edge(c1, r1, c2, r2):
        n1 = grid.get((c1, r1))
        n2 = grid.get((c2, r2))
        if not n1 or not n2:
            return None
        return edge_by_nodes.get((n1, n2))

    routes = {}

    for col in range(G):
        segs = [edge(col, row, col, row + 1) for row in range(G - 1)]
        if all(segs):
            routes[f"r_NS_{col}"] = " ".join(segs)
        else:
            segs_rev = [edge(col, row + 1, col, row) for row in range(G - 2, -1, -1)]
            if all(segs_rev):
                routes[f"r_NS_{col}"] = " ".join(segs_rev)

    for col in range(G):
        segs = [edge(col, row, col, row - 1) for row in range(G - 1, 0, -1)]
        if all(segs):
            routes[f"r_SN_{col}"] = " ".join(segs)

    for row in range(G):
        segs = [edge(col, row, col + 1, row) for col in range(G - 1)]
        if all(segs):
            routes[f"r_EW_{row}"] = " ".join(segs)
        else:
            segs_rev = [edge(col + 1, row, col, row) for col in range(G - 2, -1, -1)]
            if all(segs_rev):
                routes[f"r_EW_{row}"] = " ".join(segs_rev)

    for row in range(G):
        segs = [edge(col, row, col - 1, row) for col in range(G - 1, 0, -1)]
        if all(segs):
            routes[f"r_WE_{row}"] = " ".join(segs)

    ev_segs = [
        edge(0, 0, 1, 0), edge(1, 0, 2, 0),
        edge(2, 0, 2, 1), edge(2, 1, 2, 2),
        edge(2, 2, 3, 2), edge(3, 2, 3, 3),
    ]
    if all(ev_segs):
        routes["r_EV"] = " ".join(ev_segs)
    else:
        print(f"  WARNING: EV route incomplete: {ev_segs}")

    return routes


def _build_flow_xml(routes):
    """
    KEY FIX: emit all entries sorted by departure time so SUMO's
    incremental loader never skips a vehicle.

    Order: begin=0 flows → EV_001 (depart=900) → begin=1800 flows → truck
    """
    G = _GRID_SIZE
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
        "",
        '    <vType id="car"       accel="2.6" decel="4.5" sigma="0.5" length="4.5"  minGap="2.5" maxSpeed="13.89" speedFactor="normc(1,0.1,0.8,1.2)" color="0.3,0.6,1.0"/>',
        '    <vType id="truck"     accel="1.3" decel="3.5" sigma="0.3" length="10.0" minGap="4.0" maxSpeed="10.0"  color="0.7,0.4,0.1"/>',
        '    <vType id="emergency" accel="4.0" decel="6.0" sigma="0.1" length="6.5"  minGap="1.5" maxSpeed="20.0"  color="1.0,0.0,0.0" guiShape="emergency"/>',
        "",
    ]

    # --- Route definitions (no departure time, order doesn't matter) ---
    for route_id, edges_str in routes.items():
        lines.append(f'    <route id="{route_id}" edges="{edges_str}"/>')
    lines.append("")

    # --- Collect all timed entries as (depart_time, xml_string) tuples ---
    timed_entries = []

    flow_id = 0

    # begin=0 flows
    for col in range(G):
        for direction in ("NS", "SN"):
            rid = f"r_{direction}_{col}"
            if rid not in routes:
                continue
            timed_entries.append((0,
                f'    <flow id="f{flow_id}" type="car" route="{rid}"'
                f' begin="0"    end="1800" period="6" departLane="best" departSpeed="avg"/>'))
            flow_id += 1
            # the begin=1800 twin — added later
            timed_entries.append((1800,
                f'    <flow id="f{flow_id}" type="car" route="{rid}"'
                f' begin="1800" end="3600" period="9" departLane="best" departSpeed="avg"/>'))
            flow_id += 1

    for row in range(G):
        for direction in ("EW", "WE"):
            rid = f"r_{direction}_{row}"
            if rid not in routes:
                continue
            timed_entries.append((0,
                f'    <flow id="f{flow_id}" type="car" route="{rid}"'
                f' begin="0"    end="1800" period="6" departLane="best" departSpeed="avg"/>'))
            flow_id += 1
            timed_entries.append((1800,
                f'    <flow id="f{flow_id}" type="car" route="{rid}"'
                f' begin="1800" end="3600" period="9" departLane="best" departSpeed="avg"/>'))
            flow_id += 1

    # Truck (begin=0)
    timed_entries.append((0,
        f'    <flow id="f{flow_id}" type="truck" route="r_NS_2"'
        f' begin="0" end="3600" period="30" departLane="0" departSpeed="avg"/>'))

    # EV departs at t=900 — placed between the begin=0 and begin=1800 blocks
    ev_route = "r_EV" if "r_EV" in routes else next(iter(routes))
    timed_entries.append((900,
        f'    <vehicle id="EV_001" type="emergency" route="{ev_route}"'
        f' depart="900" departLane="0" departSpeed="max" color="1,0,0"/>'))

    # Sort strictly by departure time
    timed_entries.sort(key=lambda x: x[0])

    for _, xml_line in timed_entries:
        lines.append(xml_line)

    lines += ["", "</routes>"]
    return lines


def _write_node_map(grid):
    hospital = grid.get((3, 3), "D3")
    ev_start = grid.get((0, 0), "A0")

    lines = [f"HOSPITAL_NODE={hospital}", f"EV_START_NODE={ev_start}"]
    for row in range(_GRID_SIZE):
        for col in range(_GRID_SIZE):
            lines.append(f"NODE_{col}_{row}={grid.get((col, row), '')}")

    _NODE_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Node map written: {_NODE_FILE}")
    print(f"  Hospital: {hospital}  |  EV start: {ev_start}")


def main():
    if not _NET_FILE.exists():
        raise FileNotFoundError(
            f"Network file not found: {_NET_FILE}\n"
            "Run generate_network.py first."
        )

    print(f"Parsing network: {_NET_FILE}")
    root = ET.parse(_NET_FILE).getroot()

    edge_by_nodes = _parse_edges(root)
    grid          = _parse_grid(root)

    print(f"  {len(edge_by_nodes)} edges, {len(grid)} traffic-light junctions\n")
    print("Grid layout (row 0 = south):")
    for row in range(_GRID_SIZE):
        print("  " + "  ".join(f"{grid.get((col, row), '?'):>4}" for col in range(_GRID_SIZE)))

    routes = _build_routes(grid, edge_by_nodes)
    print(f"\nBuilt {len(routes)} routes")

    xml_lines = _build_flow_xml(routes)
    _FLOW_FILE.write_text("\n".join(xml_lines), encoding="utf-8")
    print(f"  Flow file written: {_FLOW_FILE}")

    _write_node_map(grid)

    print("\nDone. Run the simulation with:")
    print("  python scripts/traciloop.py --gui")


if __name__ == "__main__":
    main()