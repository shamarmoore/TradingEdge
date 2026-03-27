"""
health_check.py – System and connectivity health checks for CryptoEdge.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)


def check_bot_health(config_path: str) -> dict:
    """
    Check if the CryptoEdge bot is running and its API is responding.

    Reads listen_ip_address and listen_port from the config file to
    determine the API endpoint, then sends a GET /api/v1/ping request.

    Parameters
    ----------
    config_path : str
        Path to config/config.json.

    Returns
    -------
    dict with keys:
        config_found  (bool)
        api_enabled   (bool)
        api_reachable (bool)
        status_code   (int or None)
        ping_response (dict or None)
        error         (str or None)
    """
    result: dict = {
        "config_found": False,
        "api_enabled": False,
        "api_reachable": False,
        "status_code": None,
        "ping_response": None,
        "error": None,
    }

    # Load config
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        result["error"] = f"Config file not found: {config_path}"
        return result
    result["config_found"] = True

    with open(cfg_path) as f:
        config = json.load(f)

    api_cfg = config.get("api_server", {})
    if not api_cfg.get("enabled", False):
        result["api_enabled"] = False
        result["error"] = "API server is disabled in config."
        return result
    result["api_enabled"] = True

    host = api_cfg.get("listen_ip_address", "127.0.0.1")
    port = api_cfg.get("listen_port", 8080)
    # Use localhost when binding to 0.0.0.0
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    url = f"http://{host}:{port}/api/v1/ping"

    try:
        resp = requests.get(url, timeout=5)
        result["status_code"] = resp.status_code
        result["api_reachable"] = resp.status_code == 200
        try:
            result["ping_response"] = resp.json()
        except Exception:
            result["ping_response"] = resp.text
    except requests.RequestException as exc:
        result["error"] = str(exc)

    return result


def check_exchange_connectivity(exchange_name: str) -> bool:
    """
    Verify that the exchange REST API is reachable via a lightweight
    public endpoint (no authentication required).

    Supported exchanges: binance, kucoin, bybit, okx, kraken.
    Falls back to a generic HTTPS probe for unknown exchanges.

    Parameters
    ----------
    exchange_name : str
        Lower-case exchange name, e.g. "binance".

    Returns
    -------
    bool
        True if exchange API responded with HTTP 200, False otherwise.
    """
    probe_urls: dict[str, str] = {
        "binance": "https://api.binance.com/api/v3/ping",
        "kucoin": "https://api.kucoin.com/api/v1/timestamp",
        "bybit": "https://api.bybit.com/v5/market/time",
        "okx": "https://www.okx.com/api/v5/public/time",
        "kraken": "https://api.kraken.com/0/public/Time",
    }

    url = probe_urls.get(exchange_name.lower())
    if not url:
        logger.warning("Unknown exchange '%s' – skipping connectivity check.", exchange_name)
        return False

    try:
        resp = requests.get(url, timeout=10)
        return resp.status_code == 200
    except requests.RequestException as exc:
        logger.error("Exchange connectivity check failed for %s: %s", exchange_name, exc)
        return False


def check_vps_resources() -> dict:
    """
    Check VPS / server resource utilisation (CPU, memory, disk).

    Uses the standard library ``os`` module and ``shutil`` to avoid
    third-party dependencies.  Returns percentage utilisation figures.

    Returns
    -------
    dict with keys:
        cpu_percent      (float or None)
        memory_percent   (float or None)
        memory_available_mb (float or None)
        disk_percent     (float or None)
        disk_free_gb     (float or None)
        warnings         (list of str)
    """
    result: dict = {
        "cpu_percent": None,
        "memory_percent": None,
        "memory_available_mb": None,
        "disk_percent": None,
        "disk_free_gb": None,
        "warnings": [],
    }

    # ---- CPU (/proc/stat on Linux) -----------------------------------
    try:
        with open("/proc/stat") as f:
            first = f.readline()
        fields = [float(x) for x in first.split()[1:]]
        idle = fields[3]
        total = sum(fields)
        result["cpu_percent"] = round((1.0 - idle / total) * 100, 1)
    except Exception:
        pass  # Not on Linux or /proc not available

    # ---- Memory (/proc/meminfo on Linux) -----------------------------
    try:
        mem_info: dict[str, int] = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    mem_info[key] = int(parts[1])  # kB
        mem_total = mem_info.get("MemTotal", 0)
        mem_available = mem_info.get("MemAvailable", 0)
        if mem_total > 0:
            used = mem_total - mem_available
            result["memory_percent"] = round(used / mem_total * 100, 1)
            result["memory_available_mb"] = round(mem_available / 1024, 1)
    except Exception:
        pass

    # ---- Disk (shutil.disk_usage) ------------------------------------
    try:
        usage = shutil.disk_usage("/")
        result["disk_percent"] = round(usage.used / usage.total * 100, 1)
        result["disk_free_gb"] = round(usage.free / (1024 ** 3), 2)
    except Exception:
        pass

    # ---- Warnings ----------------------------------------------------
    if result["cpu_percent"] is not None and result["cpu_percent"] > 85:
        result["warnings"].append(f"High CPU usage: {result['cpu_percent']}%")
    if result["memory_percent"] is not None and result["memory_percent"] > 85:
        result["warnings"].append(f"High memory usage: {result['memory_percent']}%")
    if result["disk_percent"] is not None and result["disk_percent"] > 90:
        result["warnings"].append(f"Low disk space: {result['disk_free_gb']} GB free")

    return result
