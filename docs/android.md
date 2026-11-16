# ironasu Android hybrid (development preview)

This is a **native Kotlin + Jetpack Compose app**, not a WebView.
Android app code: `android/`. PC bridge: `ironasu/mobile_agents.py`.

## Build

- Android Studio with Android SDK 35, Java 17, Gradle 8.9
- Open the `android/` project or run `gradle -p android :app:assembleDebug`.
- APK output: `android/app/build/outputs/apk/debug/app-debug.apk`.
- GitHub Actions `Android debug APK` builds the same APK and uploads an artifact.
- No Gradle wrapper binaries are committed; a local Gradle 8.9 installation is required.

## PC-side setup

Install ironasu and start LM Studio locally on port 1234 (or set a compatible backend).
In PowerShell:

```powershell
$env:IRONASU_BACKEND_URL = "http://127.0.0.1:1234/v1"
$env:IRONASU_HOST = "0.0.0.0"
$env:IRONASU_PORT = "8765"
$env:IRONASU_SERVER_KEY = "<choose-a-long-random-secret>"
python -m ironasu
```

Find the PC's **private LAN IPv4 address**, enter `http://<PC-IP>:8765` in the Android Connect page
and type the server key. The key stays in app memory only (not saved to disk);
the URL may be edited at any time. Use a trusted private network only:
HTTP does **not** encrypt bearer tokens or corpus content. Prefer a TLS reverse proxy
for access beyond a controlled LAN. Never forward the gateway port directly to the internet.

The model-management API controls installed PC providers. Android does not expose arbitrary shell execution.
Remote harness jobs default to **disabled**, and when enabled they are **read-only**:

```powershell
$env:IRONASU_MOBILE_HARNESS_ENABLED = "1"
$env:IRONASU_MOBILE_HARNESS_WORKSPACE = "C:\\path\\to\\trusted\\repository"
```

Restart the gateway after changing environment variables. Never enable a harness on an untrusted shared network.
Mobile harness jobs use `allow_writes=False` and `allow_tests=False`, with fixed workspace and an 8-step limit.
The harness is not an OS sandbox. Treat remote model prompts and generated output as untrusted.

## Agent model slots

`workers` is in 1..16 (logical sessions). `partitions` is 4 or 16.
`model_slots` is automatically clamped to the PC's `IRONASU_MAX_CONCURRENCY`
(default 2). **Sixteen logical agents do not imply 16 concurrent GPU requests.**
Only two background mobile jobs can run at once. The resulting job ID is process-local;
cancel is cooperative and cannot stop an in-flight HTTP request immediately.
Results include 128 source evidence references at most, with truncation indicated.

## On-device inference

Offline chat uses **MediaPipe LLM Inference 0.10.27** and a compatible `.task` model
imported through Android's system file picker. The model is copied into
private application storage and loaded on the device. It is **not** bundled in the APK
and may require a high-end phone and substantial RAM/storage.
LM Studio GGUF models and Android `.task` models are different formats.
This legacy MediaPipe API is maintenance-only; migration to LiteRT-LM is recommended
once dependency/toolchain compatibility is verified. Runtime success and performance
must be checked on physical devices; emulator support can be unreliable.

## Security and limitations

- Explicit HTTPS URLs are supported. Plain HTTP is restricted by the client UI
  to private IPv4 / loopback hosts; network-level TLS is still preferable.
- Only known PC API paths are called; the server demands Bearer authentication on
  non-loopback listeners.
- Android settings, imported model weights, and chat session are local. No server
  key is written to preferences.
- Not yet validated on real Android hardware; APK build must pass CI before distribution.
- This APK is an engineering preview, not a signed production release.
