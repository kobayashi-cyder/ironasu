# Native GGUF model installation — ironasu Studio v1.4

**LM Studio / Ollama are NOT required to download and store GGUF model files.**
ironasu Studio downloads public GGUF files directly from Hugging Face into
its own persistent model folder. The Studio Web UI, metadata inspection,
installed-file listing and progress dashboard run locally.

## Quick start

1. Start ironasu Studio (`ironasu-studio --port 8766`), open the local UI.
2. Click **＋ モデルを追加**, then choose
   **ironasuへ直接保存（外部アプリ不要）**.
3. Enter a public Hugging Face model repository ID such as
   `ggml-org/gemma-3-1b-it-GGUF`; the UI can also search Hub model names.
4. Click **ファイル一覧** to display available GGUF files and byte sizes.
5. Select a quantization/file matching your disk, RAM and available compute
   resources (for example a Q4_K_M GGUF, if offered by that repository).
6. Click **ダウンロードを開始** and observe progress. The file is only
   considered installed after the expected SHA-256 has been verified.
7. Find the saved absolute filename under **保存済みモデル**. Use
   **パスをコピー** to supply it to a GGUF-compatible inference backend.

No LM Studio server is required for steps 1–7. New downloads require an
Internet connection to Hugging Face; previously downloaded files remain local
and do not require Internet to list, remove, or use in a local inference engine.

### Save location

- **Windows:** `%LOCALAPPDATA%\ironasu\models\<owner>\<repo>\<commit>\<file.gguf>`
- **Linux:** `~/.local/share/ironasu/models/<owner>/<repo>/<commit>/<file.gguf>`
- **macOS:** `~/.local/share/ironasu/models/<owner>/<repo>/<commit>/<file.gguf>`

Override the model root with `IRONASU_MODELS_DIR` **before** launching the
Studio server. This is an administrator-controlled environment setting;
the browser cannot submit an arbitrary local path.

The `installed.json` file records verified files in the root folder.
Downloads in progress are stored as `.part` files. A matching retry can resume
an interrupted partial transfer using HTTP byte ranges. The revision is pinned
to a concrete model commit and the expected LFS SHA-256 is checked before the
final file is committed. If a server ignores a range request, ironasu restarts
that file instead of appending duplicate bytes. Corrupt files are not installed.
Each download has a finite maximum size (200 GiB) and available disk space is
checked before transfer. **This is not a replacement for regular disk quotas.**

### Download cancellation and deletion

Use **中断** beside the in-progress native job. The partial file remains on disk
so the same revision may resume. Installed files are removed only after an
explicit confirmation; deletion removes the file and updates ironasu's index,
not model directories managed by LM Studio or Ollama.

## Limits and separation from inference

Native installation in v1.4 supports **public, GGUF, SHA-256-addressed LFS
artifacts only**. Gated/private repositories, unverified files, arbitrary file
types, safetensors multi-shard model families, tokenizer sidecars and models
larger than the configured limit are **not supported**. A failed or unavailable
metadata lookup cannot be bypassed by copying an arbitrary file URL.

Downloading GGUF **does not automatically load it into RAM/VRAM and is not a
promise that ironasu can execute it by itself**. Use llama.cpp, LM Studio,
Ollama or another compatible model runner to serve inference. A user-supplied
GGUF may not be executable on your hardware and may require a runtime update.
The installed-file view shows the local absolute path specifically so it can
be passed to a compatible runtime.

The original **LM Studio / Ollama delegated installer** options remain
available. Those options place models in the respective runtime's own storage
and are independent from ironasu's native folder.

### Network and security

- Native model discovery and retrieval only contact the official
  `https://huggingface.co` API or a trusted Hugging Face CDN redirect.
- Browser requests cannot supply an arbitrary remote host or download
  destination, and filenames are validated against traversal.
- No script or model-supplied Python is executed during download.
- The installed directory is persistent even if ironasu Studio itself is
  upgraded or removed. Remove model files separately to free disk space.
- Model licenses differ; review each repository's model card before use.

Hugging Face reference:
- https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download
- https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api

Run the stdlib-only regression tests with:

```sh
python -m unittest tests.test_native_models -v
```
