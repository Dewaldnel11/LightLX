import os

from .providers import (
    DEFAULT_LMSTUDIO,
    DEFAULT_OLLAMA,
    CustomOpenAI,
    LMStudio,
    Ollama,
    probe_lmstudio,
    probe_ollama,
)
from .sessions import age, list_sessions


def _dim(s):
    return f"\033[2m{s}\033[0m"


def _bold(s):
    return f"\033[1m{s}\033[0m"


def _ask(label="›"):
    return input("\n" + _bold(label) + " ").strip()


def _clean(p):
    p = p.strip()
    if p.startswith("@"):
        p = p[1:].strip()
    p = p.strip("'\"").strip().replace("\\ ", " ")
    return os.path.expanduser(p)


def pick_remote_model(title, models, url):
    print(f"\n{title}")
    if not models:
        print(_dim(f"  no models at {url}"))
        try:
            raw = _ask()
        except (EOFError, KeyboardInterrupt):
            return None
        return raw or None
    for i, name in enumerate(models, 1):
        print(f"  {i}  {name}")
    print(_dim("\nPick a number or type a model name.  (q to go back)"))
    while True:
        try:
            raw = _ask()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw.lower() in ("q", "quit", "back", ""):
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(models):
            return models[int(raw) - 1]
        return raw


def pick_custom():
    print(_dim("\nOpenAI-compatible server (vLLM, llama.cpp, …)"))
    try:
        url = _ask("base url ›")
    except (EOFError, KeyboardInterrupt):
        return None
    if not url or url.lower() in ("q", "quit"):
        return None
    if "://" not in url:
        url = "http://" + url
    try:
        model = _ask("model  ›")
        key = _ask("api key (Enter to skip) ›")
    except (EOFError, KeyboardInterrupt):
        return None
    if not model:
        return None
    return {"kind": "openai", "model": model, "url": url.rstrip("/"), "api_key": key or None}


def pick_source(state, is_model_dir, pick_local):
    ollama_url = state["prefs"].get("ollama_url") or DEFAULT_OLLAMA
    lm_url = state["prefs"].get("lmstudio_url") or DEFAULT_LMSTUDIO
    lm_key = state["prefs"].get("lmstudio_api_key") or None
    ollama = probe_ollama(ollama_url)
    lmstudio = probe_lmstudio(lm_url, api_key=lm_key)
    recents = []
    for s in state.get("recent_sources", []):
        if not s.get("kind") or not s.get("key"):
            continue
        if s["kind"] == "mlx" and not is_model_dir(s["key"]):
            continue
        recents.append(s)
    if not recents:
        recents = [{"kind": "mlx", "key": p, "label": os.path.basename(p.rstrip("/"))}
                   for p in state.get("recent_models", []) if is_model_dir(p)]

    items = []
    print()
    sessions = list_sessions(5)
    if sessions:
        print("Resume")
        for s in sessions:
            items.append(("resume", s))
            title = (s.get("title") or s.get("id") or "session")[:36]
            meta = f"{s.get('provider') or s.get('kind') or '?'} · {age(s.get('updated'))}"
            print(f"  {len(items)}  {title:<28} {_dim(meta)}")
        print()
    if recents:
        print("Recent")
        for s in recents:
            items.append(("recent", s))
            print(f"  {len(items)}  {s.get('label') or s['key']:<28} {_dim(s['kind'])}")
        print()

    print("Connect")
    items.append(("mlx", None))
    print(f"  {len(items)}  Local MLX folder           {_dim('stream or resident')}")
    items.append(("ollama", ollama))
    if ollama:
        n = len(ollama["models"])
        print(f"  {len(items)}  Ollama                     {_dim('running · ' + str(n) + ' models')}")
    else:
        print(f"  {len(items)}  Ollama                     {_dim('not running · ' + ollama_url)}")
    items.append(("lmstudio", lmstudio))
    if lmstudio and lmstudio.get("auth_required") and not lmstudio.get("models"):
        print(f"  {len(items)}  LM Studio                  {_dim('running · needs API token')}")
    elif lmstudio:
        n = len(lmstudio.get("models") or [])
        print(f"  {len(items)}  LM Studio                  {_dim('running · ' + str(n) + ' models')}")
    else:
        print(f"  {len(items)}  LM Studio                  {_dim('not running · ' + lm_url)}")
    items.append(("openai", None))
    print(f"  {len(items)}  Custom OpenAI URL          {_dim('vLLM, llama.cpp, …')}")
    print(_dim("\nPick a number — or drag in / paste a model folder.  (q to quit)"))

    while True:
        try:
            raw = _ask()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw.lower() in ("q", "quit", "exit"):
            return None
        if not raw:
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(items):
            kind, extra = items[int(raw) - 1]
            if kind == "recent":
                return _from_recent(extra, ollama_url, lm_url)
            if kind == "resume":
                return {"kind": "resume", "record": extra}
            if kind == "mlx":
                path = pick_local()
                if path:
                    return {"kind": "mlx", "path": path}
                continue
            if kind == "ollama":
                picked = _pick_ollama(ollama, ollama_url)
                if picked:
                    return picked
                continue
            if kind == "lmstudio":
                picked = _pick_lmstudio(lmstudio, lm_url, state)
                if picked:
                    return picked
                continue
            if kind == "openai":
                picked = pick_custom()
                if picked:
                    return picked
                continue
        d = _clean(raw)
        if is_model_dir(d):
            return {"kind": "mlx", "path": d}
        print(_dim(f"  not a model folder: {d}  — pick a number, paste a valid path, or q"))


def _from_recent(s, ollama_url, lm_url):
    kind = s["kind"]
    if kind == "mlx":
        return {"kind": "mlx", "path": s["key"]}
    url = s.get("url") or (ollama_url if kind == "ollama" else lm_url if kind == "lmstudio" else None)
    return {"kind": kind, "model": s["key"], "url": url, "api_key": s.get("api_key")}


def _offline_server(kind, url, hint):
    print(_dim(f"\n  {kind} is not running at {url}"))
    print(_dim(f"  {hint}"))
    print("   1  try a different address")
    print("   2  type a model name anyway")
    print(_dim("   Enter to go back"))
    try:
        c = input("  " + _bold("›") + " ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if c == "1":
        try:
            raw = input(_dim("  address › ")).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not raw:
            return None
        if "://" not in raw:
            raw = "http://" + raw
        return raw.rstrip("/"), None
    if c == "2":
        try:
            name = input(_dim("  model name › ")).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if not name:
            return None
        return url, name
    return None


def _pick_ollama(info, url):
    if info is None:
        got = _offline_server("Ollama", url, "start it with:  ollama serve")
        if got is None:
            return None
        new_url, name = got
        if name:
            return {"kind": "ollama", "model": name, "url": new_url}
        info = probe_ollama(new_url)
        url = new_url
        if info is None:
            print(_dim(f"  still not running at {url}"))
            return None
    name = pick_remote_model("Ollama models", info["models"], url)
    if not name:
        return None
    return {"kind": "ollama", "model": name, "url": info["url"]}


def _ask_lm_token(state):
    print(_dim("  LM Studio wants an API token  (Developer → Server Settings → Manage Tokens)"))
    print("   1  paste a token")
    print("   2  try again without a token")
    print(_dim("   Enter to go back"))
    try:
        c = input("  " + _bold("›") + " ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None
    if c == "1":
        try:
            token = input(_dim("  token › ")).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if token:
            state["prefs"]["lmstudio_api_key"] = token
            return token
        return None
    if c == "2":
        return ""
    return None


def _pick_lmstudio(info, url, state=None):
    state = state or {"prefs": {}}
    key = (info or {}).get("api_key") or state.get("prefs", {}).get("lmstudio_api_key") or None
    if info and info.get("auth_required") and not info.get("models"):
        token = _ask_lm_token(state)
        if token is None:
            return None
        info = probe_lmstudio(url, api_key=token or None)
        if info is None:
            print(_dim("  still cannot reach LM Studio"))
            return None
        if info.get("auth_required") and not info.get("models"):
            print(_dim("  still unauthorized — check the token, or turn auth off in LM Studio"))
            return None
        key = info.get("api_key") or token or key
    if info is None:
        got = _offline_server("LM Studio", url, "Developer → Start Server  (default port 1234)")
        if got is None:
            return None
        new_url, name = got
        if name:
            return {"kind": "lmstudio", "model": name, "url": new_url, "api_key": key}
        info = probe_lmstudio(new_url, api_key=key)
        url = new_url
        if info is None:
            print(_dim(f"  still not running at {url}"))
            return None
        key = info.get("api_key") or key
    name = pick_remote_model("LM Studio models", info.get("models") or [], info.get("url") or url)
    if not name:
        return None
    return {"kind": "lmstudio", "model": name, "url": info.get("url") or url, "api_key": info.get("api_key") or key}


def build_provider(source):
    kind = source["kind"]
    if kind == "ollama":
        return Ollama(source["model"], base_url=source.get("url"))
    if kind == "lmstudio":
        return LMStudio(source["model"], base_url=source.get("url"), api_key=source.get("api_key"))
    if kind == "openai":
        return CustomOpenAI(source["model"], source["url"], api_key=source.get("api_key"))
    raise ValueError(f"not a remote source: {kind}")
