# file-chat

Local chat UI for Ollama models (default **deepseek-r1:32b** — set the `CHAT_MODEL` and `EMBED_MODEL` env vars to use others). Point it at a folder (resume, notes, PDFs) and ask questions about those files.

## Run

```powershell
cd C:\Users\matth\file-chat
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000, paste a folder path, click **Index folder**, then chat.

Ollama must already be running with your chat model and `nomic-embed-text`
(defaults: `deepseek-r1:32b`).

Supported files: `.txt`, `.md`, `.pdf`, `.docx`, `.csv`, `.json`. Folders like `.git` and `.venv` are skipped.

The 32B model is large. If it is split across CPU and GPU, the first answer can take a minute or more because DeepSeek-R1 also “thinks” before it replies.
