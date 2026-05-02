#!/usr/bin/env python3
"""
generate_network.py
====================
Run this ONCE to generate grid6x6.net.xml using SUMO's netgenerate tool.
netgenerate produces a fully valid net.xml with all required attributes
(lane shapes, connections, dir, speed, tlLogic, internal edges, etc.)

Usage:
    python scripts/generate_network.py

Requires: SUMO installed and on PATH (netgenerate command available)
"""

import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()
NET_FILE    = PROJECT_DIR / "network" / "grid6x6.net.xml"

def run():
    NET_FILE.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "netgenerate",
        "--grid",
        "--grid.number=6",           # 6x6 grid
        "--grid.length=300",         # 300m between intersections
        "--grid.attach-length=100",  # stub roads at grid border
        "--default.lanenumber=2",    # 2 lanes per direction
        "--default.speed=13.89",     # 50 km/h
        "--tls.guess=true",          # auto-assign traffic lights
        "--tls.default-type=static", # fixed-time TLS
        "--tls.green.time=30",       # 30s green phase
        "--tls.yellow.time=3",       # 3s yellow phase
        "--turn-lanes=0",            # no dedicated turn lanes (keep it clean)
        "--output-file", str(NET_FILE),
        "--no-warnings",
    ]

    print("Running netgenerate...")
    print("  " + " ".join(cmd))
    print()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"SUCCESS: Network written to {NET_FILE}")
            print(f"  File size: {NET_FILE.stat().st_size / 1024:.0f} KB")
        else:
            print("FAILED. netgenerate output:")
            print(result.stdout)
            print(result.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print("ERROR: netgenerate not found on PATH.")
        print("Make sure SUMO is installed and its bin directory is in PATH.")
        print("Typical Windows path: C:\\Program Files (x86)\\Eclipse\\Sumo\\bin")
        sys.exit(1)

    # Verify the output is a valid SUMO net
    import xml.etree.ElementTree as ET
    try:
        tree = ET.parse(NET_FILE)
        root = tree.getroot()
        edges  = [e for e in root.findall(".//edge") if not e.get("id","").startswith(":")]
        lanes  = root.findall(".//lane")
        conns  = root.findall(".//connection")
        tls    = root.findall(".//tlLogic")
        junctions = root.findall(".//junction")
        print()
        print("Network validation:")
        print(f"  Junctions : {len(junctions)}")
        print(f"  Edges     : {len(edges)}")
        print(f"  Lanes     : {len(lanes)}")
        print(f"  Connections: {len(conns)}")
        print(f"  TLS       : {len(tls)}")
        print()
        print("Ready to simulate. Run:")
        print("  python scripts/traciloop.py --gui")
    except Exception as e:
        print(f"WARNING: Could not parse output: {e}")

if __name__ == "__main__":
    run()
