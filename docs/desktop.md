# Legacy Tk Desktop (v1.6.1)

The v1.7.0 default GUI is the native **PySide6/Qt6** window. For the
current Windows interface and installation instructions, use
[Qt6 desktop documentation](qt-desktop.md).

This older guide documents the **optional Tk fallback** only.
The Tk UI does not open a browser and can still be launched manually:

```powershell
cd C:\Users\hazim\ironasu
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\ironasu-studio-tk.exe --provider lmstudio
```

The legacy Tk fallback requires Tcl/Tk in the local Python environment.
It is not the default GUI shipped by the Windows v1.7 installer.

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
Each partition is processed as adaptive subsegments of at most 8000
characters (around 4096 near dense issue/heading markers), with exact-duplicate
single-flight reuse across workers and bounded hierarchical map/reduce.
The original text is split **losslessly**; model summaries remain lossy and
must be checked against the original. The UI and JSON export show actual
model calls, duplicate segment requests avoided and a source-position/SHA-256
evidence ledger. Details: [Parallel corpus optimisation](corpus-optimization.md).

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
