# Persistent state for LightLX — remembers recent models and preferences across
# sessions. Stored at ~/.lightlx/state.json (survives repo moves / reinstalls).

import json
import os

STATE_DIR = os.path.expanduser("~/.lightlx")
STATE_FILE = os.path.join(STATE_DIR, "state.json")

DEFAULTS = {
    "recent_models": [],
    "recent_sources": [],
    "prefs": {
        "think": False,
        "max_tokens": 512,
        "agent_max_tokens": 0,
        "temperature": 0.2,
        "ollama_url": "",
        "lmstudio_url": "",
        "lmstudio_api_key": "",
        "workspace": "",
    },
}


def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
        return {
            "recent_models": list(s.get("recent_models", [])),
            "recent_sources": list(s.get("recent_sources", [])),
            "prefs": {**DEFAULTS["prefs"], **s.get("prefs", {})},
        }
    except Exception:
        return {
            "recent_models": [],
            "recent_sources": [],
            "prefs": dict(DEFAULTS["prefs"]),
        }


def save_state(state: dict) -> None:
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass  # never let a state-write failure interrupt the user


def add_recent(state: dict, path: str, keep: int = 6) -> None:
    rec = [p for p in state.get("recent_models", []) if p != path]
    rec.insert(0, path)
    state["recent_models"] = rec[:keep]


def add_recent_source(state: dict, kind: str, key: str, label: str = None, url: str = None, keep: int = 8) -> None:
    item = {"kind": kind, "key": key, "label": label or key}
    if url:
        item["url"] = url
    rec = [s for s in state.get("recent_sources", [])
           if not (s.get("kind") == kind and s.get("key") == key)]
    rec.insert(0, item)
    state["recent_sources"] = rec[:keep]
    if kind == "mlx":
        add_recent(state, key)
