import os
import json
import logging
from typing import List

from vector_search_service import build_indexes

logging.basicConfig(level=logging.INFO)


def load_msts_from_env() -> List[str]:
    v = os.environ.get('MSTS')
    if v:
        return [x.strip() for x in v.split(',') if x.strip()]
    # fallback sample
    return [
        '253527',  # sample MST from provided examples
        '266351',
        '261457',
    ]


if __name__ == '__main__':
    msts = load_msts_from_env()
    out_dir = os.environ.get('OUT_DIR', 'faiss_indexes')
    res = build_indexes(msts, out_dir)
    print(json.dumps(res, ensure_ascii=False, indent=2))
