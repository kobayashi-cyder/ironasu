# ironasu Studio v1.9.0 — standalone PySide6 / Qt6 desktop UI

Version 1.9 makes the **PySide6/Qt6 native desktop app** the default for
`ironasu-studio` and `ironasu-desktop`. No Chrome, Edge, embedded WebView,
local HTTP UI server, CDN, or Tk widgets are used by the Qt entrypoint.

## Windows installation

Use the **Windows x64 installer or portable ZIP** built by the official
[Windows packaging workflow](https://github.com/kobayashi-cyder/ironasu/actions/workflows/windows-installer.yml).
The installer packages Python, PySide6 Essentials, Qt6 Windows plugins and required
support files. Python and PySide6 do **not** need to be separately installed
when using this EXE.

Extract the entire portable ZIP before running `ironasu-studio.exe`.
Do not execute the EXE directly from within the ZIP.

The installer is unsigned: avoid disabling security protections merely to
install it; verify release artifacts through their GitHub Actions provenance.
The app works offline, but it needs a compatible local model inference server
for LM Studio/Ollama/Jan, or an installed llama-server binary for standalone
GGUF execution.

### From source

```powershell
cd C:\Users\hazim\ironasu
git switch main
git pull --ff-only origin main
.\.venv\Scripts\python.exe -m pip install -e ".[desktop]"
.\.venv\Scripts\ironasu-studio.exe --provider lmstudio
```

You can also run `python -m ironasu.qt_desktop --provider lmstudio`.

The previously released Tk interface is still available on request:
`ironasu-studio-tk`. The browser gateway is still available explicitly as
`ironasu-studio-web`. Neither of those legacy entrypoints is launched by the
new Qt application.

The universal wheel keeps Qt as an **optional extra** so installing the
lightweight coding harness and inference core does not require GUI libraries.
If starting from the wheel rather than the self-contained Windows EXE, use
`pip install "ironasu[desktop]"`.

## Visual and accessibility improvements

- A resizable three-panel layout with `QSplitter` (navigation, central
  workspace, and scrollable Runtime inspector). Sidebar/inspector widths can
  be changed by dragging the boundaries.
- Qt6 Windows per-monitor DPI behavior; no hardcoded Tk scaling.
- Native Qt widgets, rounded dark QSS surfaces, approximately 36–40px control
  heights, generous padding, higher-contrast headings, and no legacy Tk
  notebook/combobox controls.
- UI font selection prefers installed Japanese-friendly fonts in this order:
  Yu Gothic UI, BIZ UDPGothic, Meiryo UI, Meiryo, Noto Sans CJK JP. A general
  system font is used when none of these is installed.
- Long message text and explanatory labels wrap. The Runtime inspector
  scrolls vertically rather than clipping controls on small screens.
- Model output is rendered as plain text, **never Qt rich-text/HTML**.
- Chat messages have selectable text; Reasoning text is separately expandable.

## Existing capabilities retained

**Chat:** local backend discovery, direct model selector, multi-session history
in memory, JSON export, streaming responses and cancellation, explicit LM
Studio auto-load when connected to the local port 1234, temperature/top-p/
max-token/system prompt controls, model-provided Reasoning display and
client-observed TTFT/tok/s metrics when the upstream supplies token usage.

**Model management:** Hugging Face verified public GGUF search/file selection/
download, installed model listing, delegated LM Studio/Ollama model download,
manual LM Studio loading, download status and cancellation, and ironasu-managed
`llama-server` GGUF launch/stop. The llama.cpp executable remains a separate
installation, and native GGUF inference is not silently assumed.

**16-agent corpus analysis:** 4/16-way lossless partitioning, up to 16
lightweight worker sessions, independent model-concurrency cap, adaptive
subchunks, exact duplicate single-flight reuse, bounded map/reduce, source
SHA-256 evidence index, summary/error display, progress, cancellation and
JSON reports. These are **not** 16 independent GPU model instances.

## Validation

- PySide6 Qt6 widgets are constructed offscreen in Linux CI, with three tabs,
  model management tree and bounded 16-agent configuration assertions.
- Windows CI runs the frozen EXE's `--self-test` and verifies the bundled
  Qt module, then builds the portable ZIP and per-user Inno Setup installer.
- The frozen Windows GUI is additionally smoke-tested where the hosted runner
  provides a display. Real Radeon hardware performance and visual behavior
  at the user's actual display scaling still require a local check.

Run from source:

```sh
python -m unittest tests.test_qt_desktop -v
QT_QPA_PLATFORM=offscreen python -m ironasu.qt_desktop --smoke-gui qt-smoke.json
```

The smoke diagnostic writes JSON without contacting a model server.

## Git worktree isolated parallel missions (v1.9)

The Mission tab has an opt-in Git worktree switch, up to 16 separate editors,
independent baseline/candidate verification and a separate explicit integration
approval button. It is off by default, and both write and test permissions
must be approved. Tests are **not** OS-sandboxed.
See [isolated missions](isolated-missions.md).

## Mission Controller

The fourth tab provides bounded, persisted Plan→Act→Verify→Repair coding missions, up to 16 agent sessions, read-only defaults, explicit write/test permission prompts, separate model-call slots, SQLite resume and test-based verification. See [Mission guide](mission.md).

## Privacy and limitations

- Qt frontend runs the inference gateway objects **in-process**; it does not
  expose the Studio frontend over TCP.
- In auto backend discovery mode, only predefined loopback servers are probed.
  Explicit remote backend URLs are supported by the CLI but may transmit
  prompts to a remote provider.
- Importing the Qt module for `--help` or `--self-test` does not load Qt.
  Actual desktop startup needs PySide6 installed in the Python environment
  unless using the bundled Windows EXE.
- No external browser framework is used; the legacy Tk and browser UIs remain
  separate supported entrypoints.
