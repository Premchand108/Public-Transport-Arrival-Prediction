"""
run_web.py
==========
Quick launcher for the PulseTransit AI Web Application.
Automatically starts the Flask server and launches your default web browser.

Usage:
    python run_web.py
    python run_web.py --port 5000
"""

import sys
import socket
import argparse
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from app import start_server


def find_available_port(start_port=5000, max_attempts=10):
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start_port


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PulseTransit AI Web Application")
    parser.add_argument("--port", type=int, default=5000, help="Port to run web app (default: 5000)")
    args = parser.parse_args()

    port = find_available_port(args.port)
    start_server(port=port)
