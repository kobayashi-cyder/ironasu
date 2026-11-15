# ironasu Studio installation and backend compatibility (v1.2)

This package installs an **offline-first local Web UI plus an inference gateway**.
It does not bundle model weights, LM Studio, Ollama, GPU drivers or any model
execution runtime. Those must be installed and their models downloaded before
you disconnect from the Internet.

## Windows 10/11 (x64): graphical installer, no Python required

1. Open the [offline installer workflow](https://github.com/kobayashi-cyder/ironasu/actions/workflows/windows-installer.yml).
2. Select a **successful** run on the main branch and download the
   `ironasu-studio-windows-x64-offline-installers` artifact.
3. Unzip that artifact and run `ironasu-studio-1.3.0-windows-x64-setup.exe`.
   The installer is per-user and adds a Start Menu shortcut. It installs
   completely offline and does not need administrator privileges.
4. Launch **ironasu Studio** from the Start Menu. A local browser tab opens
   at `http://127.0.0.1:8765/ui`.
5. Start a local model server (LM Studio, Ollama, etc.) and **load a model**.
   Model availability refreshes automatically, including servers started after
   Studio. If the browser is not available, use another locally installed browser.

The workflow also supplies a `-portable.zip` containing a standalone executable:
extract *all* files to one directory, then run `ironasu-studio.exe`.
Do not copy only the EXE from the portable ZIP: companion files are required.

**Signing:** GitHub Actions installers are currently unsigned. Windows may show
an unknown-publisher warning. Verify that the artifact comes from this
repository's successful workflow; do not disable system security to install it.

## Python installation (Windows, macOS, Linux)

Python 3.10 or newer is required for this method. The core UI/gateway has
**zero third-party runtime dependencies**.

Online installation from a trusted local checkout:

```sh
python -m pip install .
ironasu-studio
```

Offline wheel installation (wheel downloaded and copied to the PC in advance):

```sh
python -m pip install --no-index --no-deps ./ironasu-1.3.0-py3-none-any.whl
ironasu-studio --provider auto
```

Get the wheel from the `ironasu-studio-offline-python-wheel` CI artifact.
Unlike source installation, installing this ready-built wheel does **not**
require setuptools or a network connection. You can also run directly from
the source root using `python -m ironasu.studio` without installing a wheel.

The original `python -m ironasu` and `ironasu` commands retain the
manual-configuration gateway behavior; the new `ironasu-studio` entrypoint
enables local-provider discovery and browser launching.

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
