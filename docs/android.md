# ironasu Android — Hybrid native client (preview)

The Android app uses **Kotlin / Jetpack Compose**, not a WebView. It supports:

* **Remote PC inference:** authenticated `ironasu Studio` REST gateway; chat completions and models.
* **Read-only harness team:** `POST /ui/agents/start` with up to 16 tasks, scheduler jobs executed on the PC. `model_slots` is bounded by the PC's `IRONASU_MAX_CONCURRENCY`; worker count does not equal simultaneously resident LLMs.
* **Remote model administration:** list, search and download GGUF on the PC; start/stop the PC-managed llama.cpp runtime or load an installed LM Studio model.
* **On-device inference:** import an explicitly supported MediaPipe `.task` bundle using the system file picker. Models are copied to private app storage. The native Android app runs generation with MediaPipe when you toggle on-device mode. **A GGUF file cannot be imported as `.task`**. Device/model compatibility, CPU/GPU support and output quality depend on the device and model.

This is a preliminary **debug build** and not a signed production release. The model weights and the desktop `llama-server` executable are **not bundled**. No background or remote command execution is silently enabled.

## Build APK

Open `android/` in Android Studio with JDK 17 and Android SDK 36, or install Gradle 8.13 and run:

```bash
gradle -p android :app:assembleDebug
```

The APK, if the build succeeds, is at `android/app/build/outputs/apk/debug/app-debug.apk`.

GitHub Actions workflow **Android APK** builds it on push/PR and uploads the artifact `ironasu-android-debug-apk` after a successful build. No APK is checked into Git.

Dependencies are pinned: AGP 8.13.2, Kotlin/Compose compiler 2.3.0, Gradle 8.13, MediaPipe Tasks GenAI 0.10.35, Compose BOM 2025.09.00. Building requires internet for Maven/SDK packages.

## Connecting the Android app

On the **PC**, in PowerShell before launching ironasu:

```powershell
$env:IRONASU_HOST = "0.0.0.0"
$env:IRONASU_SERVER_KEY = "<a-long-random-secret>"
$env:IRONASU_MAX_CONCURRENCY = "2"
$env:IRONASU_MOBILE_WORKSPACE = "C:\Users\your-user\ironasu"
.\.venv\Scripts\python.exe -m ironasu.studio --port 8766 --no-browser
```

The key is required for non-loopback binds, by existing `Config.validate` behavior. Configure Windows Firewall to allow inbound TCP 8766 **only from a trusted LAN/VPN**. LM Studio can remain bound to PC loopback (`127.0.0.1:1234`): the Android app connects to ironasu, not directly to LM Studio.

On **Android**, enter the PC's private LAN IP, e.g. `http://192.168.1.20:8766`, and the same server key in **接続**. The IP in the app's initial field is a placeholder and should be changed. For Android emulator, use `http://10.0.2.2:8766` if the emulator can access the gateway and the gateway is configured to listen accordingly.

**Transport warning:** HTTP on a shared wireless LAN is not encrypted, so prompts and bearer tokens can be intercepted. The app refuses public HTTP hostnames, but that does not provide cryptographic protection. Prefer HTTPS via a trusted reverse proxy or private VPN. Do not expose port 8766 to the public internet. The token lives only in app memory, and the app does not back up its data.

## Mobile agent API

`IRONASU_MOBILE_WORKSPACE` is a PC administrator-controlled local folder. The API **does not accept workspace paths, arbitrary shell commands, write or test permissions** from Android. Jobs run through existing `run_harness_team` with `allow_writes=False`, `allow_tests=False`, bounded to a single active team. Outputs are stored in a small in-process history, lost after server restart.

```http
POST /ui/agents/start
Authorization: Bearer <IRONASU_SERVER_KEY>
Content-Type: application/json

{
  "workers": 4,
  "model_slots": 2,
  "shared_context": "Read-only analysis",
  "tasks": [
    {"task_id": "agent_01", "prompt": "Analyze repository architecture"},
    {"task_id": "agent_02", "prompt": "Identify performance issues", "depends_on": ["agent_01"]}
  ]
}
```

Returns `202` with `{id, state, workers, model_slots, results, ...}`. Poll `GET /ui/agents/jobs/{id}`, or inspect recent jobs with `GET /ui/agents/jobs`. `401` means missing/wrong key, `409` means another team is active, `400` means invalid request. Start and read routes are protected by the same bearer key as the existing ironasu gateway. There is intentionally **no cancellation endpoint**: cancelling a harness call can leave side effects ambiguous. In this release, no mobile-originated writes or test executions are permitted.

## Known limitations

* The Android app does not bundle a GGUF inference engine or model weights; the phone side requires a compatible `.task` bundle, sufficient free space, RAM and MediaPipe-supported hardware. Model import is not model-format conversion. A `.task` model may still fail at runtime if it does not match the MediaPipe version/device.
* Agent results arrive when a team run completes, not as individually streamed tokens. There is no background service to guarantee job observation across Android process death; the server job continues until complete and can be queried by ID.
* Running 16 agent sessions does **not** imply 16 simultaneous model contexts. Default LLM model-call concurrency remains limited to the PC's configured budget.
* This is not a release-key-signed APK. Android Studio or the GitHub Actions debug build must first successfully assemble the project.

## References

* Android Gradle plugin 8.13 compatibility: https://developer.android.com/build/releases/agp-8-13-0-release-notes
* MediaPipe GenAI LLM Android API: https://ai.google.dev/gemma/docs/conversions/hf-to-mediapipe-task
* Existing ironasu runtime instructions: `docs/runtime.md`