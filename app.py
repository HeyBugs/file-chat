from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INDEX_PATH = DATA_DIR / "index.json"
STATIC_DIR = ROOT / "static"

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "deepseek-r1:32b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

ALLOWED_SUFFIXES = {".txt", ".md", ".pdf", ".docx", ".csv", ".json", ".rtf"}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", "data"}
MAX_FILE_BYTES = 8 * 1024 * 1024
CHUNK_CHARS = 900
CHUNK_OVERLAP = 120
TOP_K = 6

app = FastAPI(title="file-chat")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class IndexRequest(BaseModel):
    folder: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[dict[str, str]] = Field(default_factory=list)


def load_index() -> dict[str, Any]:
    if not INDEX_PATH.exists():
        return {"folder": None, "files": [], "chunks": []}
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def save_index(payload: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(payload), encoding="utf-8")


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv", ".json", ".rtf"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        from docx import Document

        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


def chunk_text(text: str, source: str) -> list[dict[str, str]]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    chunks: list[dict[str, str]] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + CHUNK_CHARS)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append({"source": source, "text": piece})
        if end == len(cleaned):
            break
        start = max(0, end - CHUNK_OVERLAP)
    return chunks


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    async with httpx.AsyncClient(timeout=120) as client:
        for text in texts:
            response = await client.post(
                f"{OLLAMA}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Ollama embeddings failed: {response.text}",
                )
            vectors.append(response.json()["embedding"])
    return vectors


def retrieve(query_vec: list[float], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = []
    for chunk in chunks:
        scored.append((cosine(query_vec, chunk["embedding"]), chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for score, chunk in scored[:TOP_K] if score > 0.15]


@app.get("/")
async def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
async def status() -> dict[str, Any]:
    index = load_index()
    ollama_ok = False
    models: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{OLLAMA}/api/tags")
            ollama_ok = response.status_code == 200
            models = [item["name"] for item in response.json().get("models", [])]
    except httpx.HTTPError:
        ollama_ok = False
    return {
        "ollama": ollama_ok,
        "chat_model": CHAT_MODEL,
        "embed_model": EMBED_MODEL,
        "models": models,
        "folder": index.get("folder"),
        "files": index.get("files", []),
        "chunk_count": len(index.get("chunks", [])),
    }


@app.post("/api/index")
async def index_folder(body: IndexRequest) -> dict[str, Any]:
    folder = Path(body.folder).expanduser()
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=400, detail="That folder does not exist.")

    files: list[Path] = []
    for path in folder.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES:
            if path.stat().st_size <= MAX_FILE_BYTES:
                files.append(path)

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No supported files found (.txt, .md, .pdf, .docx, .csv, .json).",
        )

    chunks: list[dict[str, str]] = []
    relative_files: list[str] = []
    for path in files:
        rel = str(path.relative_to(folder))
        relative_files.append(rel)
        try:
            text = extract_text(path)
        except Exception as exc:  # noqa: BLE001
            relative_files[-1] = f"{rel} (skipped: {exc})"
            continue
        chunks.extend(chunk_text(text, rel))

    if not chunks:
        raise HTTPException(status_code=400, detail="Files were found, but no readable text.")

    vectors = await embed_texts([chunk["text"] for chunk in chunks])
    stored = []
    for chunk, vector in zip(chunks, vectors):
        stored.append({**chunk, "embedding": vector})

    payload = {
        "folder": str(folder.resolve()),
        "files": relative_files,
        "chunks": stored,
    }
    save_index(payload)
    return {"folder": payload["folder"], "files": relative_files, "chunk_count": len(stored)}


@app.post("/api/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    index = load_index()
    chunks = index.get("chunks") or []
    if not chunks:
        raise HTTPException(status_code=400, detail="Index a folder first.")

    query_vec = (await embed_texts([body.message]))[0]
    matches = retrieve(query_vec, chunks)
    context = "\n\n".join(
        f"[{item['source']}]\n{item['text']}" for item in matches
    ) or "No relevant file snippets were found."

    system = (
        "You are a local file assistant. Answer using the provided file excerpts. "
        "If the files do not contain the answer, say so. Quote the file name when you use it. "
        "Be concise."
    )
    messages = [{"role": "system", "content": system}]
    for turn in body.history[-8:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append(
        {
            "role": "user",
            "content": f"File excerpts:\n{context}\n\nQuestion:\n{body.message}",
        }
    )

    async def stream():
        yield f"data: {json.dumps({'status': 'generating'})}\n\n"
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{OLLAMA}/api/chat",
                    json={
                        "model": CHAT_MODEL,
                        "messages": messages,
                        "stream": True,
                        "options": {"temperature": 0.3},
                    },
                ) as response:
                    if response.status_code != 200:
                        err = await response.aread()
                        yield f"data: {json.dumps({'error': err.decode('utf-8', 'ignore')})}\n\n"
                        return
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        payload = json.loads(line)
                        piece = payload.get("message", {}).get("content", "")
                        if piece:
                            yield f"data: {json.dumps({'token': piece})}\n\n"
                        if payload.get("done"):
                            yield f"data: {json.dumps({'done': True})}\n\n"
        except httpx.HTTPError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
