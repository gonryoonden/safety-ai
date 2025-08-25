import os
import json
import time
import logging
from typing import Any, Dict

from flask import Flask, request, jsonify

from gemini_service import chat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


@app.route('/chat', methods=['POST'])
def chat_route():
    req_id = request.headers.get('X-Request-ID') or str(int(time.time() * 1000))
    t0 = time.monotonic()
    try:
        payload: Dict[str, Any] = request.get_json(force=True) or {}
        messages = payload.get('messages') or []
        logger.info("[%s] /chat messages=%d", req_id, len(messages))
        res = chat(messages)
        dt = time.monotonic() - t0
        logger.info("[%s] /chat done in %.0fms", req_id, dt * 1000)
        return jsonify({"request_id": req_id, **res})
    except Exception as e:
        dt = time.monotonic() - t0
        logger.exception("[%s] /chat error after %.0fms: %s", req_id, dt * 1000, e)
        return jsonify({"request_id": req_id, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '8080')))
