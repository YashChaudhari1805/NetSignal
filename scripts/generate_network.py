#!/usr/bin/env python3
"""
generate_network.py
===================
Generates the 6×6 intersection grid network using SUMO's ``netgenerate`` tool.

Run this once before any simulation. The output file is written to
``network/grid6x6.net.xml``.

Usage:
    python scripts/generate_network.py

Requires:
    SUMO installed with ``netgenerate`` on PATH.
    Download: https://sumo.dlr.de/docs/Downloads.php
"""

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


_PROJECT_DIR = Path(__file__).parent.parent.resolve()
_NET_FILE    = _PROJECT_DIR / "network" / "grid6x6.net.xml"

_NETGENERATE_ARGS = [
    "--grid",
    "--grid.number=6",
    "--grid.length=300",
    "--grid.attach-length=100",
    "--default.lanenumber=2",
    "--default.speed=13.89",
    "--tls.guess=true",
    "--tls.default-type=static",
    "--tls.green.time=30",
    "--tls.yellow.time=3",
    "--turn-lanes=0",
    "--no-warnings",
]


def generate() -> None:
    """
    Invoke ``netgenerate`` to produce the grid network XML.

    Exits with a non-zero code if the tool is missing or returns an error.
    """
    _NET_FILE.parent.mkdir(parents=True, exist_ok=True)

    cmd = ["netgenerate"] + _NETGENERATE_ARGS + ["--output-file", str(_NET_FILE)]

    print("Running netgenerate...")
    print("  " + " ".join(cmd) + "\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit(
            "ERROR: 'netgenerate' not found.\n"
            "Ensure SUMO is installed and its bin directory is on PATH.\n"
            "  Windows: C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\n"
            "  macOS/Linux: brew/apt install sumo then re-open your terminal."
        )

    if result.returncode != 0:
        print("netgenerate failed:")
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)

    size_kb = _NET_FILE.stat().st_size / 1024
    print(f"Network written: {_NET_FILE}  ({size_kb:.0f} KB)")


def validate() -> None:
    """
    Parse the generated network file and print a basic element count summary.

    Raises:
        SystemExit: If the file cannot be parsed.
    """
    try:
        root = ET.parse(_NET_FILE).getroot()
    except ET.ParseError as exc:
        sys.exit(f"ERROR: Could not parse generated network: {exc}")

    non_internal = [e for e in root.findall(".//edge") if not e.get("id", "").startswith(":")]

    print("\nNetwork summary:")
    print(f"  Junctions  : {len(root.findall('.//junction'))}")
    print(f"  Edges      : {len(non_internal)}")
    print(f"  Lanes      : {len(root.findall('.//lane'))}")
    print(f"  Connections: {len(root.findall('.//connection'))}")
    print(f"  TLS        : {len(root.findall('.//tlLogic'))}")
    print("\nNext step:")
    print("  python scripts/fix_flows.py")


def main() -> None:
    """Generate and validate the grid network."""
    generate()
    validate()


if __name__ == "__main__":
    main()
