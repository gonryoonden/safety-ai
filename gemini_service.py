import os
import re
import json
import logging
from typing import Any, Dict, Optional

import google.generativeai as genai

from utils import get_attachment_link

logger = logging.getLogger(__name__)


def _configure_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)


def make_model():
    _configure_gemini()
    tool = {
        "function_declarations": [
            {
                "name": "get_attachment_link",
                "description": "법령명(law_name)과 별표 번호(attachment_number)로 해당 별표 PDF URL 반환",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "law_name": {"type": "STRING"},
                        "attachment_number": {"type": "NUMBER"},
                    },
                    "required": ["law_name", "attachment_number"],
                },
            }
        ]
    }
    model = genai.GenerativeModel(model_name=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"), tools=[tool])
    return model


def maybe_extract_attachment_intent(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"별표\s*(\d+)", text)
    if not m:
        return None
    num = int(m.group(1))
    # Law name heuristic: between quotes or around the phrase
    lm = re.search(r"[\"'“”‘’]?([가-힣A-Za-z0-9·ㆍ\s]+?)[\"'“”‘’]?\s*의\s*별표", text)
    law_name = None
    if lm:
        law_name = lm.group(1).strip()
    return {"attachment_number": num, "law_name": law_name}


def chat(messages: Any) -> Dict[str, Any]:
    model = make_model()
    # Try using function call
    resp = model.generate_content(messages, tools=model.tools)
    if getattr(resp, 'candidates', None):
        cand = resp.candidates[0]
        if getattr(cand, 'content', None) and getattr(cand.content, 'parts', None):
            for p in cand.content.parts:
                if getattr(p, 'function_call', None):
                    fc = p.function_call
                    if fc.name == 'get_attachment_link':
                        args = fc.args or {}
                        law_name = args.get('law_name')
                        attachment_number = args.get('attachment_number')
                        if isinstance(attachment_number, (int, float)):
                            attachment_number = int(attachment_number)
                        if not law_name or not isinstance(attachment_number, int):
                            return {"text": "요청 파라미터가 부족합니다. 법령명과 별표 번호를 정확히 알려주세요."}
                        result = get_attachment_link(law_name, attachment_number)
                        # Re-inject tool result
                        tool_msg = {
                            "role": "tool",
                            "tool_name": "get_attachment_link",
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                        final = model.generate_content(messages + [tool_msg], tools=model.tools)
                        return {"text": final.text}
    # Fallback: simple regex to detect intent
    user_text = "".join([m.get('parts', [{}])[0].get('text', '') for m in messages if m.get('role') == 'user'])
    intent = maybe_extract_attachment_intent(user_text)
    if intent and intent.get('law_name'):
        result = get_attachment_link(intent['law_name'], intent['attachment_number'])
        if result.get('ok'):
            return {"text": f"별표 {intent['attachment_number']} PDF 링크: {result['pdf_url']}"}
        else:
            msg = result.get('message') or result.get('error') or '링크를 찾지 못했습니다.'
            if result.get('detail_page'):
                msg += f" 상세 페이지: {result['detail_page']}"
            return {"text": msg}
    # Otherwise: placeholder
    return {"text": "문의 내용을 이해했습니다. 추가로 법령명이나 세부 요청을 알려주세요."}
