# ironasu Studio Desktop v1.6 — native window and 16 corpus agents

**Default: a standalone Windows desktop app, not a browser.**

Version 1.6 changes the installed ironasu-studio.exe to a native Tk interface.
It does not open Chrome/Edge, embed a browser engine or bind TCP port 8765.
The LM Studio inference backend may still listen on port 1234.

## Windows setup

Download the successful ironasu-studio-windows-x64-offline-installers artifact
from https://github.com/kobayashi-cyder/ironasu/actions/workflows/windows-installer.yml
and run the v1.6 per-user installer inside it. Alternatively extract the
portable ZIP and run ironasu-studio.exe. Keep the complete extracted folder
(including the Tcl/Tk support files). The executable runs without separately
installing Python, Chrome, Node or WebView2.

These community CI installers are unsigned. Verify the GitHub workflow
provenance; do not disable Windows security protections to install.

From the checked-out repository:

~~~powershell
cd C:\Users\hazim\ironasu
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\ironasu-studio.exe --provider lmstudio
~~~

To keep using the legacy browser UI on purpose:
~~~powershell
.\.venv\Scripts\ironasu-studio-web.exe --provider lmstudio --port 8766
~~~

The native application does not start a localhost UI server. On Linux/macOS
the universal wheel also supports native Tk if the OS provides Tcl/Tk.

## Native GUI tabs

- Chat: session history in memory, export, model selection, Reasoning display,
  TTFT, upstream-usage-based estimated tokens/s, response cancellation,
  LM Studio manual and opt-in automatic load.
- Model management: public Hugging Face GGUF downloads, Ollama/LM Studio
  delegated downloads, installed file lists, llama-server-managed startup,
  stop, removal and progress.
- エージェント ×16: read-only parallel corpus analysis with 4/16 partitions,
  up to 16 workers, separate bounded inference calls, detailed reports,
  explicit cancellation and JSON export.

## Agent task controls: 16 workers, 4/16 chunks

1. Open the エージェント ×16 tab.
2. Paste the corpus or choose a UTF-8 .txt/.md/.csv/.json/.log file.
   Limit: 1,000,000 Unicode characters.
3. Set an analysis objective common to all workers.
4. Select 4 or 16 lossless ordered partitions and 1-16 workers.
5. Choose how many model requests can run simultaneously (1-16).
   The effective cap is min(chosen slots, IRONASU_MAX_CONCURRENCY).
6. Run. The app shows subsegment progress and produces a summary from all
   successfully completed partitions. Unsuccessful partitions are reported
   separately; it does not pretend partial output covers the entire corpus.

Splitting is ordered and lossless: concatenating the 4 or 16 partitions
recreates every original character. Boundaries prefer nearby newlines.
Each partition is processed in sequential subsegments of at most 8000
characters, with incremental compression. This avoids silently truncating
large chunks. It is NOT lossless semantic compression: LLM summaries may
omit details, and important findings must be checked against the original.

16 agents means 16 independent lightweight sessions, NOT 16 GPU model
instances. Model calls default to at most two concurrent requests, even
with 16 workers. To raise the backend concurrency deliberately, set:

~~~powershell
$env:IRONASU_MAX_CONCURRENCY = "4"
ironasu-studio --provider lmstudio
~~~

A single Radeon GPU can become slower at high parallelism, depending on
VRAM, KV cache and scheduler overhead. Benchmark at 1/2/4 call slots before
raising the limit. The older ironasu-team coding harness also supports
a separate DAG scheduler with 16 read-only tool-capable sessions; desktop
corpus analysis does not automatically execute tools, shells or file edits.

## Boundaries

- GUI startup and already-downloaded models can work offline.
- New Hugging Face model downloads require the Internet.
- GGUF execution needs a compatible local llama-server binary. It is NOT
  bundled in the Windows desktop EXE.
- LM Studio/Ollama models need their own local runtime server for inference.
- Corpus input is treated as untrusted text, not executable instructions.
- Agent outputs and chat sessions remain in memory unless JSON is exported.
- Models are not independently loaded in 16 GPU contexts, and internal KV
  caches are not shared unless the chosen backend implements this.
- The old ironasu HTTP gateway and browser UI remain available as separate
  opt-in commands and are not started by ironasu-studio.

## Smoke test

~~~sh
python -m unittest tests.test_desktop tests.test_corpus_agents -v
ironasu-studio --self-test ./desktop-check.json
~~~

CI additionally starts the real Tk widgets off-screen in the frozen
Windows executable via --smoke-gui. Hardware performance still requires
testing with the user's Radeon and selected model/runtime.
