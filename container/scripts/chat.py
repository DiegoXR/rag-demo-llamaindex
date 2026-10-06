"""A terminal chat client for the demo's contract — the job hubworld's /playground
page does, from a terminal.

It talks to FastAPI directly, so it needs the two local processes running:
  npm run dev:resources     (the Worker, lending its bindings)
  npm run dev:api           (FastAPI + LlamaIndex on :8080)
then, from the project root:  npm run chat

Not part of the image: the Dockerfile copies only app/.
"""

import json
import os
import shlex
import time
from pathlib import Path

import httpx

from app.settings import MAX_INPUT_CHARS, MAX_MESSAGES

CHAT_URL = os.environ.get("CHAT_URL", "http://localhost:8080").rstrip("/")

HELP = """Escribe una pregunta, o un comando:
  /docs                     documentos indexados
  /upload <ruta.md> [título] sube un Markdown (sin título: el nombre del archivo)
  /delete <doc_id>          borra un documento
  /reset                    empieza una conversación nueva
  /ayuda                    esta ayuda
  /salir                    termina (también Ctrl+C)"""


def trim(history: list[dict]) -> list[dict]:
    """The conversation sent to the demo: newest turns first until a contract limit
    is reached, starting on a user turn — same rule as history() in hubworld's
    +page.svelte, so a long chat keeps working instead of failing with 413."""
    out, chars = [], 0
    for turn in reversed(history):
        if len(out) == MAX_MESSAGES or chars + len(turn["content"]) > MAX_INPUT_CHARS:
            break
        out.insert(0, turn)
        chars += len(turn["content"])
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def ask(client: httpx.Client, history: list[dict], question: str) -> str | None:
    """Sends one question and prints the answer as it streams. Returns the answer,
    or None if it failed (so the caller keeps it out of the history)."""
    if len(question) > MAX_INPUT_CHARS:
        print(f"[!] La pregunta pasa de {MAX_INPUT_CHARS} caracteres.")
        return None

    messages = trim([*history, {"role": "user", "content": question}])
    started = time.monotonic()
    answer, sources, usage = [], [], None

    # Server-Sent Events: one JSON object per `data:` line.
    with client.stream("POST", "/chat", json={"messages": messages}) as response:
        if response.status_code != 200:
            response.read()
            print(f"[{response.status_code}] {response.text}")
            return None
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: "):])
            if event["type"] == "token":
                print(event["text"], end="", flush=True)
                answer.append(event["text"])
            elif event["type"] == "sources":
                sources = event["sources"]
            elif event["type"] == "done":
                usage = event["usage"]
            elif event["type"] == "error":
                print(f"\n[error] {event['message']}")
                return None

    print()
    for s in sources:
        score = f" · {s['score']}" if s["score"] is not None else " · resumen"
        print(f"  ↳ {s['title']}{score}")
    tokens = f" · {usage['input_tokens']} in / {usage['output_tokens']} out tokens" if usage else ""
    print(f"  {time.monotonic() - started:.1f} s{tokens}")
    return "".join(answer)


def show_docs(client: httpx.Client) -> None:
    docs = client.get("/documents").raise_for_status().json()["documents"]
    if not docs:
        print("No hay documentos. Sube uno con /upload <ruta.md>.")
    for d in docs:
        print(f"  {d['doc_id']}  ·  {d['title']}  ·  {d['chunks']} chunks")


def upload(client: httpx.Client, args: list[str]) -> None:
    if not args:
        print("Uso: /upload <ruta.md> [título]")
        return
    path = Path(args[0])
    title = " ".join(args[1:]) or path.stem
    response = client.post(
        "/documents",
        params={"title": title},
        content=path.read_bytes(),
        headers={"content-type": "text/markdown; charset=utf-8"},
    )
    if response.status_code != 201:
        print(f"[{response.status_code}] {response.text}")
        return
    doc = response.json()
    action = "Reemplazado" if doc["replaced"] else "Subido"
    print(f"{action}: {doc['doc_id']} ({doc['chunks']} chunks). Tarda unos segundos en ser buscable.")


def delete(client: httpx.Client, args: list[str]) -> None:
    if not args:
        print("Uso: /delete <doc_id>")
        return
    response = client.delete(f"/documents/{args[0]}")
    print(f"Borrado: {args[0]}" if response.status_code == 200 else f"[{response.status_code}] {response.text}")


def main() -> None:
    print(f"Chat con {CHAT_URL}. /ayuda para ver los comandos.")
    history: list[dict] = []
    # The answer is time-boxed at 30 s by the demo; the extra margin covers the network.
    with httpx.Client(base_url=CHAT_URL, timeout=httpx.Timeout(10, read=60)) as client:
        while True:
            try:
                line = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not line:
                continue
            try:
                if line.startswith("/"):
                    # posix=False keeps Windows backslashes; quotes allow paths with spaces.
                    command, *args = [a.strip('"') for a in shlex.split(line, posix=False)]
                    if command == "/salir":
                        return
                    elif command == "/ayuda":
                        print(HELP)
                    elif command == "/docs":
                        show_docs(client)
                    elif command == "/upload":
                        upload(client, args)
                    elif command == "/delete":
                        delete(client, args)
                    elif command == "/reset":
                        history.clear()
                        print("Conversación nueva.")
                    else:
                        print(f"Comando desconocido: {command}. /ayuda para ver los comandos.")
                    continue

                answer = ask(client, history, line)
                if answer is not None:
                    history += [{"role": "user", "content": line}, {"role": "assistant", "content": answer}]
            except httpx.ConnectError:
                print(f"[!] No hay nada escuchando en {CHAT_URL}. ¿Está corriendo `npm run dev:api`?")
            except httpx.HTTPError as e:
                print(f"[!] {type(e).__name__}: {e}")
            except KeyboardInterrupt:
                # Closing the stream makes the demo cancel the answer (cancel_run).
                print("\n[cancelado]")
            except OSError as e:
                print(f"[!] {e}")


if __name__ == "__main__":
    main()
