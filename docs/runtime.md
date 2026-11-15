# ironasu Studio v1.5 — managed GGUF inference, LM Studio autoload and inference telemetry

This release completes the path from a verified downloaded GGUF to local chat
**when a compatible `llama-server` executable is installed**. The app
launches the inference subprocess, waits for readiness, and routes chat
requests to it automatically. You do **not** need to start a separate
model-serving app manually.

Important: the Windows installer ships the Python gateway and Web UI but does
**not** bundle llama.cpp, GPU drivers, or large model weights. A
hardware-appropriate `llama-server` executable must be installed once.
Native GGUF execution is optional; LM Studio remains available.

## Quick start — Windows PowerShell

1. Get a version of llama.cpp **llama-server.exe** suitable for your CPU/GPU
   from official sources, verify its provenance, and unpack it locally.
   On Radeon devices, choose a runtime built with a compatible Vulkan or
   other supported accelerator backend. A CPU build is a fallback.
2. Tell ironasu where the binary is **before** starting Studio:

```powershell
$env:IRONASU_LLAMA_SERVER = "C:\\Tools\\llama.cpp\\llama-server.exe"
cd C:\\Users\\hazim\\ironasu
.\\.venv\\Scripts\\python.exe -m ironasu.studio --port 8766
```

You can also put `llama-server.exe` on PATH; the environment variable then
isn't needed. The binary path is administrator/user-controlled, **not a field
in a browser request**.

3. Open `http://127.0.0.1:8766/ui`, click **＋ モデルを追加**, choose
   **ironasuへ直接保存（外部アプリ不要）**, and download a verified public GGUF
   (or use an already installed GGUF).
4. In **保存済みモデル**, click **GGUFを起動** next to that file.
   You can select a context length and GPU-layer target first. Default GPU
   behavior is the llama-server runtime's own policy.
5. Wait for **管理ランタイム: running**. The selected model appears in
   the top-right model list as `ironasu-local-…`. Choose it for chat.
6. Click **管理モデルを停止** when finished to release CPU/GPU memory.

The spawned llama-server binds only to a dynamically selected
`127.0.0.1` port, disables its own web UI and uses a random API key.
ironasu does not run shell commands submitted by a browser. Only verified
models listed in ironasu's installed index are accepted. Model integrity
(SHA-256) is checked before each new launch, so very large files can take
time to validate. Startup is asynchronous; another model can be launched
after the current one has stopped. The subprocess is terminated when
ironasu Studio exits.

The model runtime is **separate from the downloader**. A GGUF download alone
does not provide a CPU/GPU inference kernel. This release automates runner
startup but does not silently install unverified executables.

Official llama.cpp server details:
https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md

## LM Studio model loading from the same interface

Select LM Studio as the active backend and use
**＋ モデルを追加 → LM Studio → インストール済みモデル → ロード**
to explicitly load an existing model. This uses LM Studio's
`POST /api/v1/models/load` management API.

If you prefer **autoload on the first chat**, enable the
**LM Studioモデルをチャット時に自動ロード（明示許可）** checkbox in the
Runtime sidebar and select an explicit LM Studio model in the header.

The checkbox is off by default. When enabled, ironasu verifies the selected
model is installed and only then loads it, immediately before sending the
chat request. Only a locally configured LM Studio server at
`127.0.0.1:1234` is accepted. Other providers and native GGUF aliases
cannot trigger LM Studio autoload.

## Timing, throughput and Reasoning display

For each completed chat, the Runtime sidebar shows:

- **TTFT:** browser-observed duration from request submission to the first
  streamed content **or model-provided reasoning** received. This includes
  local HTTP, scheduling and model work, so it is not a pure prefill benchmark.
  TTFT is **unavailable** for non-streaming responses.
- **tok/s:** upstream-reported `usage.completion_tokens` divided by the
  observed time since the first streamed output. It is a **client estimate**,
  not the exact GPU decode speed. It is blank when token usage is absent;
  ironasu will not infer fake token counts from characters.
- **Reasoning duration:** observed time between the first reasoning segment
  and the first final answer segment (or last reasoning segment). This is not
  a measurement of a model's hidden/internal computation.
- **characters/s:** plain final-answer characters divided by end-to-end
  browser latency, provided as a fallback diagnostic (not tokens/s).

If a backend sends `reasoning_content`, `reasoning`,
`reasoning_details[].text` or explicit `<think>…</think>` sections, the
UI displays them in an **expandable Reasoning panel**, separate from the
final answer. It never fabricates missing reasoning and never disables
reasoning by default. The panel contains **only upstream-provided text**.
Model-dependent protocol differences can affect visibility.

Actual LLM inference timing should be benchmarked at the backend as well as
the browser, especially when measuring TTFT/throughput with sparse SSE
events, backpressure or long local model loads.

## Operational and privacy notes

- Default listener is loopback. Don't expose your unauthenticated local
  inference or management API to the public Internet.
- Existing LM Studio / Ollama / vLLM / LocalAI / Jan proxy compatibility and
  original `python -m ironasu` entrypoint remain available.
- Installing a model is distinct from automatically accepting its license,
  memory requirements or safety. Choose GPU-offload and context limits based
  on local system capabilities.
- The managed runtime reports a readiness error if no suitable
  `llama-server` executable is installed; this is not a silent fallback to
  downloading or executing another program.
- Runtime API: `GET /ui/runtime/status`, `POST /ui/runtime/start`,
  `POST /ui/runtime/stop`. Same-origin checks and configured server-key
  authentication apply. API bodies specify **an installed index ID only**,
  never arbitrary file paths or executables.

See [native model installation](native-install.md) for verified downloads and
[Studio installation](install.md) for portable/installer/wheel packaging.
