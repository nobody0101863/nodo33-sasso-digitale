import os
from typing import Dict

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class CodexMessage(BaseModel):
    message: str


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/codex")
async def codex_endpoint(payload: CodexMessage) -> Dict[str, str]:
    return {"message": payload.message}


def main() -> None:
    """Entry point used when running `python server.py`."""
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
