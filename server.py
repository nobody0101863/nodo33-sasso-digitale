# ╔════════════════════════════════════════════════════════════╗
# ║           SECURE SERVER - NODO33 SASSO DIGITALE            ║
# ║  Includes: Rate Limiting, Auth, Logging, Memory, Filter    ║
# ║  Motto: "La luce non si vende. La si regala."              ║
# ╚════════════════════════════════════════════════════════════╝

import os
import re
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.cors import CORSMiddleware
from cryptography.fernet import Fernet

# ════════════════════════════════════════════════════════════════
# SECURITY SETTINGS
# ════════════════════════════════════════════════════════════════

API_KEY = os.getenv("API_KEY", "changeme")
FERNET_KEY = os.getenv("FERNET_KEY")
fernet = Fernet(FERNET_KEY.encode()) if FERNET_KEY else None

# ════════════════════════════════════════════════════════════════
# APP INITIALIZATION
# ════════════════════════════════════════════════════════════════

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Secure Codex Server",
    description="Nodo33 - Sasso Digitale | La luce non si vende. La si regala.",
    version="2.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )


# ════════════════════════════════════════════════════════════════
# LOGGING CONFIG
# ════════════════════════════════════════════════════════════════

logger = logging.getLogger("secure_server")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
))
logger.addHandler(handler)

# ════════════════════════════════════════════════════════════════
# MEMORY LOGGER (HASH + ENCRYPTION OPTIONAL)
# ════════════════════════════════════════════════════════════════


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def encrypt_text(text: str) -> str:
    if fernet:
        return fernet.encrypt(text.encode()).decode()
    return text


def anonymize_prompt(prompt: str) -> str:
    if not prompt:
        return "[EMPTY]"
    return prompt[:10] + "..." + prompt[-10:] if len(prompt) > 25 else prompt


def secure_log_memory(
    provider: str,
    question: str,
    answer: str,
    tag: Optional[str] = None
) -> None:
    timestamp = datetime.utcnow().isoformat()
    log_data = {
        "timestamp": timestamp,
        "provider": provider,
        "prompt_hash": hash_text(question),
        "answer_preview": anonymize_prompt(answer),
        "tag": tag or "llm",
    }
    if fernet:
        log_data["encrypted_question"] = encrypt_text(question)
        log_data["encrypted_answer"] = encrypt_text(answer)
    logger.info(f"[MEMORY_LOG] {json.dumps(log_data)}")


# ════════════════════════════════════════════════════════════════
# LLM HANDLER (MOCKED + MODERATION + SANITIZATION)
# ════════════════════════════════════════════════════════════════

INJECTION_PATTERNS = re.compile(
    r"(ignore previous|act as|system prompt|sudo|token|password|key)",
    re.IGNORECASE
)

MODERATION_PATTERNS = re.compile(
    r"(kill|bomb|nazi|racist|sexist|hate|terror)",
    re.IGNORECASE
)


def secure_llm_handler(
    prompt: str,
    model: str = "mock-model",
    temperature: float = 0.7
) -> dict:
    if INJECTION_PATTERNS.search(prompt):
        raise HTTPException(
            status_code=400,
            detail="Prompt injection attempt detected"
        )

    response = f"[SECURE RESPONSE] {prompt[:100]}..."

    if MODERATION_PATTERNS.search(response):
        raise HTTPException(
            status_code=403,
            detail="Content blocked by moderation filter"
        )

    return {
        "answer": response,
        "model": model,
        "tokens_used": len(prompt.split()),
        "timestamp": datetime.utcnow().isoformat()
    }


# ════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ════════════════════════════════════════════════════════════════


class CodexMessage(BaseModel):
    message: str


class LLMRequest(BaseModel):
    question: str
    temperature: float = 0.7
    system_prompt: str = ""


# ════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════


@app.get("/health")
async def health_check() -> Dict[str, str | bool]:
    return {"status": "ok", "secure": True, "audit": True}


@app.post("/codex")
@limiter.limit("10/minute")
async def codex_endpoint(
    payload: CodexMessage,
    request: Request,
    api_key: str = Depends(verify_api_key)
) -> Dict[str, str]:
    logger.info(json.dumps({
        "event": "codex_message",
        "ip": get_remote_address(request),
        "timestamp": datetime.utcnow().isoformat()
    }))
    return {"message": payload.message}


@app.post("/api/llm/{provider}")
@limiter.limit("5/minute")
async def call_llm(
    provider: str,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    try:
        payload = await request.json()
        question = payload.get("question", "").strip()
        temperature = float(payload.get("temperature", 0.7))
        system_prompt = payload.get("system_prompt", "").strip()

        if not question:
            raise HTTPException(status_code=400, detail="Missing question")

        full_prompt = f"{system_prompt}\n{question}".strip()

        response = secure_llm_handler(
            full_prompt,
            model=provider,
            temperature=temperature
        )

        secure_log_memory(
            provider=provider,
            question=question,
            answer=response["answer"],
            tag="llm_query"
        )
        logger.info(json.dumps({
            "event": "llm_query",
            "provider": provider,
            "tokens_used": response["tokens_used"],
            "timestamp": response["timestamp"]
        }))

        return JSONResponse(content=response)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(json.dumps({"event": "error", "error": str(e)}))
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# ════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════


def main() -> None:
    """Entry point used when running `python server.py`."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
