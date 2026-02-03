import argparse
import json
import os
import time
from typing import List

from utils import LawAPIClient, resolve_law_name


def _parse_names(arg: str) -> List[str]:
    if not arg:
        return []
    # law names can contain spaces; split only by comma
    parts = [p.strip() for p in arg.split(",")]
    return [p for p in parts if p]


def _load_registry(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {"updated_at": None, "laws": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "laws" in data:
            return data
    except Exception:
        pass
    return {"updated_at": None, "laws": []}


def _save_registry(path: str, registry: dict) -> None:
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def _update_registry(registry: dict, entry: dict) -> None:
    laws = registry.get("laws") or []
    name = entry.get("law_name")
    updated = False
    for i, it in enumerate(laws):
        if it.get("law_name") == name:
            laws[i] = {**it, **entry}
            updated = True
            break
    if not updated:
        laws.append(entry)
    registry["laws"] = laws
    registry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--law-names", required=True, help="Comma/space separated law names")
    ap.add_argument("--mst-registry", default="mst_registry.json")
    args = ap.parse_args()

    names = _parse_names(args.law_names)
    if not names:
        print("[error] no law names provided")
        return 2

    registry = _load_registry(args.mst_registry)
    client = LawAPIClient()
    results = []
    for name in names:
        resolved = resolve_law_name(client, name)
        resolved["resolved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _update_registry(registry, resolved)
        results.append(resolved)

    _save_registry(args.mst_registry, registry)
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
