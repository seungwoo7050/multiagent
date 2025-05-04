# src/api/middleware/json_serialization_middleware.py
import json
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from src.utils.serialization import serialize_to_json

class JSONResponseMiddleware(BaseHTTPMiddleware):
    """
    • 라우터에서 반환한 파이썬 객체(dict, list, BaseModel 등)를
      모두 일관된 JSON 문자열로 직렬화해서 내려줌.
    • 기존 FastAPI의 JSONResponse를 쓰지 않을 때(PlainTextResponse 등)
      자동 포맷팅 보장.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)

        # 이미 JSONResponse라면 건너뛰기
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            return response

        # 바디를 파이썬 객체로 꺼내보기
        body = None
        try:
            # FastAPI가 Pydantic 모델을 반환할 때
            # `response.body` 가 비어있고 `.background` 로 처리될 수 있어,
            # 가능하다면 `response.__dict__['body_obj']` 같은 내부 속성을 쓰실 수도 있습니다.
            body = json.loads(response.body.decode())
        except Exception:
            # JSON으로 파싱할 수 없는 경우엔 건너뛰기
            return response

        # serialize_to_json → 문자열, JSONResponse → 올바른 JSON 헤더+바이트
        json_str = serialize_to_json(body)
        return JSONResponse(content=json.loads(json_str), status_code=response.status_code)
