# ironasu Studio installation and backend compatibility (v1.7.0)

**Default desktop: PySide6 / Qt6 native Windows application.** The program
runs in its own independent window; Chrome/Edge and a locally served browser
UI are not required. The model *inference* engine (LM Studio, Ollama or an
installed llama-server binary for standalone GGUF) remains separate.

## Windows x64: graphical installer and portable ZIP

1. Go to the [Windows installer CI workflow](https://github.com/kobayashi-cyder/ironasu/actions/workflows/windows-installer.yml).
2. Select the successful latest **main** run and download the
   `ironasu-studio-windows-x64-offline-installers` artifact.
3. Extract the downloaded artifact ZIP. Run
   `ironasu-studio-1.7.0-windows-x64-setup.exe` or extract the inner
   `ironasu-studio-1.7.0-windows-x64-portable.zip` and launch
   `ironasu-studio.exe`.
4. The new Qt6 application opens a native independent window with Chat,
   Model management and エージェント ×16 tabs. No browser URL is involved.
5. Start a model inference backend (LM Studio is on 127.0.0.1:1234 by
   default), then select its model in the Runtime sidebar.

The Windows EXE bundles Python and Qt6 support files, so there is no need
to separately install either. Keep all files from the extracted portable
ZIP. The community-built installer is unsigned: verify GitHub Actions
provenance and do not disable Windows security.

## Python source or wheel installation

For the *Qt desktop*, install the optional GUI extra:

```powershell
cd C:\Users\hazim\ironasu
git switch main
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m pip install -e ".[desktop]"
.\.venv\Scripts\ironasu-studio.exe --provider lmstudio
```

The universal Python wheel deliberately keeps the model gateway/harness
dependency-light. Installing it without the `desktop` extra supports core
tools and CLI self-test but not actual Qt window startup. The Windows installer
**does** contain PySide6/Qt6.

Legacy alternatives are still opt-in:
- `ironasu-studio-tk`: older independent Tk window;
- `ironasu-studio-web --provider lmstudio --port 8766`: browser-based UI;
- `python -m ironasu`: original local inference gateway.

For GPU inference of directly downloaded GGUF models, install a compatible
`llama-server` binary separately. Model downloads alone do not supply a
GPU runtime. See [Qt6 desktop details](qt-desktop.md),
[managed GGUF runner](runtime.md) and
[direct model downloads](native-install.md).

## Supported local OpenAI-compatible endpoints

| Provider | Preset | Typical local OpenAI endpoint |
| --- | --- | --- |
| LM Studio | `--provider lmstudio` | `http://127.0.0.1:1234/v1` |
| Ollama | `--provider ollama` | `http://127.0.0.1:11434/v1` |
| llama.cpp server | `--provider llamacpp` | `http://127.0.0.1:8080/v1` |
| LocalAI | `--provider localai` | `http://127.0.0.1:8080/v1` |
| vLLM | `--provider vllm` | `http://127.0.0.1:8000/v1` |
| Jan | `--provider jan` | `http://127.0.0.1:1337/v1` |

`--provider auto` (default) probes only the listed loopback addresses. On
a shared port 8080, the active server is treated as an OpenAI-compatible API;
there is no way to reliably distinguish llama.cpp from LocalAI based on
the port alone. A working `GET /v1/models` endpoint is required for automatic
model selection. If model discovery is unavailable, explicitly supply model IDs.

Compatibility targets the overlapping **/v1/models** and
**/v1/chat/completions** text/API protocol surface, including SSE streaming if
the server supports it. This does **not** mean identical sampling options,
reasoning behavior, multimodal support, tools, load/unload controls or vendor
APIs. Capability differences remain the responsibility of the chosen runtime.

### Explicit endpoint and model IDs

```powershell
ironasu-studio --provider lmstudio
ironasu-studio --provider ollama
ironasu-studio --provider auto --doctor
ironasu-studio --backend-url http://127.0.0.1:8080/v1 --quality-model my-actual-model
ironasu-studio --provider ollama --fast-model small-model-id --quality-model large-model-id
```

When either route has no explicit model ID, the **first returned model** is
used for that route; if there is only one model, "fast" and "quality" will
necessarily use the same model. No claim is made that the first ID is the
fastest or highest-quality model. Specify both IDs to use two different models.

`--doctor` only checks model discovery, prints diagnostic JSON, and exits.
The `--no-browser` option disables automatic tab opening for headless use.
For external backends, `--backend-url` is opt-in and may transmit your prompts
and chat content outside your computer.

## Known limits

- Automatic discovery searches only loopback server ports, not the local
  network or the Internet. The UI does not download or install LLMs.
- API compatibility does not include Ollama's native `/api/chat` or LM Studio's
  proprietary model-management REST API.
- The Windows installer is x64-only. Build from source or install the universal
  Python wheel on other supported systems.
- SSE passthrough does not cache responses; it relies on upstream streaming.
- Your browser remains the UI renderer. This is **not** a native Windows
  WebView implementation.
- Offline installation means no network is required for **installation and UI
  startup** after copying the package; inference still requires a functional
  local model server and model files.

For UI details and privacy behavior, see [Studio](studio.md).
