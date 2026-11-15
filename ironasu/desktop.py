"""ironasu Studio Desktop: native Tk controls, no embedded or external browser.

The local model gateway and managers run in-process. This entry point never
starts an HTTP listener and never calls webbrowser.open. It works entirely
offline once the selected local backend and models are installed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Callable

from .compat import AutoHTTPBackend, PRESETS
from .corpus_agents import CorpusAgentTeam, MAX_CORPUS_CHARS, split_corpus, split_segments
from .core import AdaptiveEngine, Config, HTTPBackend, InferenceError
from .desktop_logic import GenerationStats, SSEEvents, extract_reasoning, extract_text, normalized_model_selection
from .models import ModelManager
from .native_models import NativeModelManager
from .runtime import ManagedGGUFRuntime, RuntimeBackendMux
from .studio import configure, parser as studio_parser

BACKGROUND = "#0e1219"
NAV = "#121924"
SURFACE = "#1a2430"
SURFACE_ALT = "#202d38"
BORDER = "#354755"
FOREGROUND = "#ebf4f4"
MUTED = "#99abb9"
MINT = "#95dfbd"
WARN = "#ffb3a7"


@dataclass
class Session:
    title: str = "新しいチャット"
    messages: list[dict[str, Any]] = field(default_factory=list)


def desktop_parser() -> argparse.ArgumentParser:
    parser = studio_parser()
    parser.prog = "ironasu-desktop"
    parser.description = "Native ironasu Studio window (no browser or web server)"
    parser.add_argument("--self-test", metavar="FILE",
                        help="Write a JSON environment check without launching a window")
    parser.add_argument("--smoke-gui", metavar="FILE",
                        help="Create the real native window off-screen and write a diagnostic report")
    parser.add_argument("--version-desktop", action="version", version="ironasu Desktop 1.6.0")
    return parser


def desktop_self_test(path: str) -> None:
    """Frozen-build smoke test. No network or GUI display is needed."""
    from importlib.util import find_spec
    report = {
        "mode": "native-tk-desktop",
        "browser_required": False,
        "http_server_required": False,
        "tkinter_available": find_spec("tkinter") is not None,
        "studio_version": "1.6.0",
    }
    target = Path(path).expanduser()
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")


class DesktopWindow:
    """Tk widgets are created only when launched, not during CLI/test imports."""
    def __init__(self, root: Any, config: Config, backend: Any,
                 *, provider: str = "auto") -> None:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        from tkinter.scrolledtext import ScrolledText
        self.tk, self.ttk = tk, ttk
        self.filedialog, self.messagebox, self.ScrolledText = filedialog, messagebox, ScrolledText
        self.root = root
        self.config = config
        self.provider = provider
        self.native = NativeModelManager()
        self.managed = ManagedGGUFRuntime(self.native)
        self.model_manager = ModelManager(config)
        self.upstream = backend
        self.mux = RuntimeBackendMux(self.upstream, self.managed)
        self.engine = AdaptiveEngine(config, self.mux)
        self.tasks: queue.Queue[tuple] = queue.Queue()
        self.cancel_event = threading.Event()
        self.active_stream = None
        self.stream_lock = threading.Lock()
        self.busy = False
        self.closed = False
        self.sessions: list[Session] = [Session()]
        self.session_index = 0
        self.last_status = ""
        self.agent_cancel = threading.Event()
        self.agents_running = False
        self.corpus_file_text: str | None = None
        self.corpus_file_path = ""
        self.agent_report = None
        self.agent_total_segments = 0
        self.last_model_id = None
        self.available_models: list[str] = []
        self.native_files: list[dict[str, Any]] = []
        self.native_results: list[dict[str, Any]] = []
        self.native_installed: list[dict[str, Any]] = []
        self.delegated_installed: list[dict[str, Any]] = []
        self._jobs_cache = ""
        self._last_library_refresh = 0.0
        self._last_connection_refresh = 0.0
        self._updating_provider = False
        self._build_theme()
        self._build()
        self._render_sessions()
        self._render_conversation()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(80, self._drain)
        self.root.after(120, self.refresh_connection)
        self.root.after(450, self.refresh_library)
        self.root.after(500, self._tick)
        self.prompt.focus_set()

    def _build_theme(self) -> None:
        tk, ttk = self.tk, self.ttk
        self.root.title("ironasu Studio · Desktop")
        self.root.geometry("1320x840")
        self.root.minsize(940, 590)
        self.root.configure(bg=BACKGROUND)
        self.root.option_add("*Font", "{Segoe UI} 10")
        self.root.option_add("*insertBackground", FOREGROUND)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("Studio.TCombobox", fieldbackground=SURFACE_ALT,
                        background=SURFACE_ALT, foreground=FOREGROUND,
                        arrowcolor=MINT, padding=(8, 7))
        style.map("Studio.TCombobox", foreground=[("readonly", FOREGROUND)],
                  fieldbackground=[("readonly", SURFACE_ALT)])
        style.configure("Studio.Treeview", background=SURFACE,
                        fieldbackground=SURFACE, foreground=FOREGROUND,
                        bordercolor=BORDER, rowheight=28)
        style.configure("Studio.Treeview.Heading", background=SURFACE_ALT,
                        foreground=MINT, font=("Segoe UI", 9, "bold"))
        style.map("Studio.Treeview", background=[("selected", "#2c5346")],
                  foreground=[("selected", FOREGROUND)])
        style.configure("Studio.TProgressbar", background=MINT, troughcolor=SURFACE_ALT)
        style.configure("Studio.TNotebook", background=BACKGROUND, borderwidth=0)
        style.configure("Studio.TNotebook.Tab", background=NAV,
                        foreground=MUTED, padding=(17, 11))
        style.map("Studio.TNotebook.Tab", foreground=[("selected", MINT)],
                  background=[("selected", SURFACE)])
        style.configure("Studio.TCheckbutton", background=SURFACE,
                        foreground=FOREGROUND)
        style.map("Studio.TCheckbutton", background=[("active", SURFACE)])

    def _label(self, parent: Any, text: str, *, color: str = FOREGROUND,
               size: int = 10, bold: bool = False, **kw: Any) -> Any:
        font = ("Segoe UI", size, "bold" if bold else "normal")
        return self.tk.Label(parent, text=text, bg=kw.pop("bg", parent.cget("bg")),
                             fg=color, font=font, anchor=kw.pop("anchor", "w"), **kw)

    def _button(self, parent: Any, text: str, command: Callable,
                *, strong: bool = False, width: int | None = None) -> Any:
        background = MINT if strong else SURFACE_ALT
        fg = "#133024" if strong else FOREGROUND
        button = self.tk.Button(parent, text=text, command=command,
                                bg=background, fg=fg, activebackground="#b6f0d2" if strong else BORDER,
                                activeforeground="#153527" if strong else FOREGROUND,
                                relief="flat", cursor="hand2", padx=13, pady=8,
                                font=("Segoe UI", 9, "bold" if strong else "normal"),
                                highlightthickness=0, bd=0)
        if width:
            button.config(width=width)
        return button

    def _entry(self, parent: Any, variable: Any = None, *, width: int = 22) -> Any:
        return self.tk.Entry(parent, textvariable=variable, bg=SURFACE_ALT,
                             fg=FOREGROUND, insertbackground=MINT,
                             relief="flat", bd=0, width=width,
                             font=("Segoe UI", 10), highlightthickness=1,
                             highlightbackground=BORDER, highlightcolor=MINT)

    def _combobox(self, parent: Any, variable: Any, values: list[str],
                  width: int = 22) -> Any:
        return self.ttk.Combobox(parent, textvariable=variable, values=values,
                                 state="readonly", width=width, style="Studio.TCombobox")

    def _section(self, parent: Any, heading: str) -> Any:
        outer = self.tk.Frame(parent, bg=SURFACE, padx=13, pady=12)
        self._label(outer, heading, size=11, bold=True, color=MINT).pack(anchor="w", pady=(0, 9))
        return outer

    def _build(self) -> None:
        tk, ttk = self.tk, self.ttk
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.sidebar = tk.Frame(self.root, bg=NAV, width=228, padx=13, pady=20)
        self.sidebar.grid(row=0, column=0, sticky="ns")
        self.sidebar.grid_propagate(False)
        self._label(self.sidebar, "◈   ironasu", color=MINT, size=19, bold=True).pack(anchor="w", pady=(0, 3))
        self._label(self.sidebar, "      STUDIO  •  DESKTOP", color=MUTED, size=8).pack(anchor="w", pady=(0, 24))
        self._button(self.sidebar, "+ 新しいチャット", self.new_session, strong=True).pack(fill="x", pady=(0, 22))
        self._label(self.sidebar, "セッション", color=MUTED, size=9, bold=True).pack(anchor="w", pady=(0, 8))
        self.session_list = tk.Listbox(self.sidebar, bg=NAV, fg=FOREGROUND,
                                       selectbackground="#284839", selectforeground=FOREGROUND,
                                       highlightthickness=0, borderwidth=0,
                                       activestyle="none", font=("Segoe UI", 10),
                                       exportselection=False)
        self.session_list.pack(fill="both", expand=True)
        self.session_list.bind("<<ListboxSelect>>", self._choose_session)
        self._button(self.sidebar, "チャットをJSON保存", self.export_session).pack(fill="x", pady=(15, 6))
        self._button(self.sidebar, "この会話を削除", self.delete_session).pack(fill="x")
        self._label(self.sidebar, "履歴は起動中のメモリのみ", color=MUTED, size=8).pack(
            anchor="w", pady=(17, 0))

        main = tk.Frame(self.root, bg=BACKGROUND, padx=14, pady=10)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)
        self.tabs = ttk.Notebook(main, style="Studio.TNotebook")
        self.tabs.grid(row=0, column=0, sticky="nsew")
        self.chat_tab = tk.Frame(self.tabs, bg=BACKGROUND)
        self.library_tab = tk.Frame(self.tabs, bg=BACKGROUND)
        self.agent_tab = tk.Frame(self.tabs, bg=BACKGROUND)
        self.tabs.add(self.chat_tab, text="  チャット  ")
        self.tabs.add(self.library_tab, text="  モデル管理  ")
        self.tabs.add(self.agent_tab, text="  エージェント ×16  ")
        self._build_chat()
        self._build_library()
        self._build_agents()

    def _build_chat(self) -> None:
        tk, ttk = self.tk, self.ttk
        page = self.chat_tab
        page.grid_rowconfigure(1, weight=1)
        page.grid_columnconfigure(0, weight=1)
        header = tk.Frame(page, bg=BACKGROUND, pady=11)
        header.grid(row=0, column=0, sticky="ew")
        self.chat_title = self._label(header, "新しいチャット", size=15, bold=True)
        self.chat_title.pack(side="left")
        self.status_label = self._label(header, "モデル確認中", color=MUTED, size=9)
        self.status_label.pack(side="right", padx=10)
        self.transcript = self.ScrolledText(
            page, wrap="word", bg=BACKGROUND, fg=FOREGROUND,
            insertbackground=MINT, borderwidth=0, relief="flat",
            padx=23, pady=19, font=("Segoe UI", 11), state="disabled",
            selectbackground="#3c5d54",
        )
        self.transcript.grid(row=1, column=0, sticky="nsew")
        self.transcript.tag_configure("user_title", foreground="#91bed5",
                                      font=("Segoe UI", 10, "bold"), spacing1=16)
        self.transcript.tag_configure("assistant_title", foreground=MINT,
                                      font=("Segoe UI", 10, "bold"), spacing1=16)
        self.transcript.tag_configure("body", foreground=FOREGROUND,
                                      spacing1=7, spacing3=8, lmargin1=7, lmargin2=7)
        self.transcript.tag_configure("thinking_title", foreground="#9fbeb0",
                                      font=("Segoe UI", 9, "bold"), spacing1=8)
        self.transcript.tag_configure("thinking", foreground="#9cabb7", lmargin1=15,
                                      lmargin2=15, font=("Segoe UI", 10))
        self.transcript.tag_configure("status", foreground=MUTED,
                                      font=("Segoe UI", 9), spacing3=16)
        composer = tk.Frame(page, bg=SURFACE, padx=12, pady=10)
        composer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        composer.grid_columnconfigure(0, weight=1)
        self.prompt = tk.Text(composer, wrap="word", height=4, bg=SURFACE,
                              fg=FOREGROUND, insertbackground=MINT,
                              borderwidth=0, padx=7, pady=5,
                              font=("Segoe UI", 10), undo=True)
        self.prompt.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.prompt.bind("<Control-Return>", self._send_key)
        self.prompt.bind("<Shift-Return>", self._send_key)
        self.composer_note = self._label(
            composer, "Ctrl+Enterで送信  •  Reasoningはモデルから提供された範囲のみ表示",
            color=MUTED, size=8, bg=SURFACE)
        self.composer_note.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.stop_button = self._button(composer, "■ 停止", self.stop_generation)
        self.stop_button.grid(row=1, column=1, padx=8, pady=(8, 0))
        self.stop_button.configure(state="disabled")
        self.send_button = self._button(composer, "送信 ↑", self.send, strong=True)
        self.send_button.grid(row=1, column=2, pady=(8, 0))

        right = tk.Frame(self.root, width=308, bg=SURFACE, padx=15, pady=17)
        right.grid(row=0, column=2, sticky="ns")
        right.grid_propagate(False)
        self._label(right, "Runtime", size=14, bold=True, bg=SURFACE).pack(anchor="w", pady=(0, 14))
        self._label(right, "接続先", color=MUTED, size=9, bg=SURFACE).pack(anchor="w")
        self.provider_var = tk.StringVar(value=self.provider)
        self.provider_combo = self._combobox(
            right, self.provider_var, ["auto", *PRESETS], width=28)
        self.provider_combo.pack(fill="x", pady=(5, 7))
        self.provider_combo.bind("<<ComboboxSelected>>", self._change_provider)
        self._button(right, "接続とモデル一覧を更新", self.refresh_connection).pack(fill="x", pady=(0, 13))
        self._label(right, "モデル / ルート", color=MUTED, size=9, bg=SURFACE).pack(anchor="w")
        self.model_var = tk.StringVar(value="auto")
        self.model_combo = self._combobox(
            right, self.model_var, ["auto", "fast", "quality"], width=28)
        self.model_combo.pack(fill="x", pady=(5, 13))
        self.autoload_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(right, text="LM Studioモデルを自動ロード",
                        variable=self.autoload_var, style="Studio.TCheckbutton").pack(
                            anchor="w", pady=(0, 14))
        self._label(right, "Temperature", bg=SURFACE, color=MUTED, size=9).pack(anchor="w")
        self.temperature_var = tk.DoubleVar(value=0.7)
        tk.Scale(right, from_=0, to=1.5, resolution=0.05, orient="horizontal",
                 variable=self.temperature_var, bg=SURFACE, fg=FOREGROUND,
                 troughcolor=SURFACE_ALT, activebackground=MINT, highlightthickness=0,
                 showvalue=True).pack(fill="x")
        self._label(right, "Top P", bg=SURFACE, color=MUTED, size=9).pack(anchor="w")
        self.top_p_var = tk.DoubleVar(value=0.95)
        tk.Scale(right, from_=0.05, to=1, resolution=0.05, orient="horizontal",
                 variable=self.top_p_var, bg=SURFACE, fg=FOREGROUND,
                 troughcolor=SURFACE_ALT, activebackground=MINT, highlightthickness=0,
                 showvalue=True).pack(fill="x")
        self._label(right, "最大生成トークン", bg=SURFACE, color=MUTED, size=9).pack(
            anchor="w", pady=(6, 0))
        self.tokens_var = tk.IntVar(value=1024)
        self._entry(right, self.tokens_var, width=24).pack(fill="x", pady=(5, 12), ipady=5)
        self._label(right, "System prompt", bg=SURFACE, color=MUTED, size=9).pack(anchor="w")
        self.system_prompt = tk.Text(right, height=4, wrap="word", bg=SURFACE_ALT,
                                     fg=FOREGROUND, insertbackground=MINT,
                                     relief="flat", padx=8, pady=7)
        self.system_prompt.pack(fill="x", pady=(5, 8))
        self.show_reasoning = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text="Reasoningを別枠表示",
                        variable=self.show_reasoning,
                        style="Studio.TCheckbutton",
                        command=self._render_conversation).pack(anchor="w", pady=(0, 8))
        self.stream_enabled = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text="ストリーミング",
                        variable=self.stream_enabled,
                        style="Studio.TCheckbutton").pack(anchor="w", pady=(0, 12))
        self._label(right, "推論計測（端末観測）", size=10, bold=True,
                    color=MINT, bg=SURFACE).pack(anchor="w", pady=(8, 8))
        self.metrics_label = self._label(
            right, "TTFT: —\n生成速度: —\nReasoning受信: —\n文字/秒: —",
            color=FOREGROUND, bg=SURFACE, size=10, justify="left")
        self.metrics_label.pack(fill="x", pady=(0, 8))
        self._label(right, "TTFTは最初の受信まで。tok/sは上流が\n"
                    "token usageを返す場合のみ算出します。",
                    color=MUTED, size=8, bg=SURFACE, justify="left").pack(anchor="w")
        self.load_note = self._label(
            right, "稼働中: —  /  待機中: —", color=MUTED, size=8, bg=SURFACE)
        self.load_note.pack(anchor="w", pady=(18, 0))

    def _build_library(self) -> None:
        tk, ttk = self.tk, self.ttk
        page = self.library_tab
        page.grid_columnconfigure(0, weight=1)
        page.grid_columnconfigure(1, weight=1)
        page.grid_rowconfigure(2, weight=1)
        heading = tk.Frame(page, bg=BACKGROUND, pady=12)
        heading.grid(row=0, column=0, columnspan=2, sticky="ew")
        self._label(heading, "モデル管理", size=15, bold=True).pack(side="left")
        self._label(heading, "ダウンロード後の推論にはGGUF対応ランタイムが必要",
                    size=8, color=MUTED).pack(side="right")
        native = self._section(page, "01  Hugging Face → ironasuに直接保存")
        native.grid(row=1, column=0, sticky="nsew", padx=(0, 7), pady=(0, 10))
        native.grid_columnconfigure(0, weight=1)
        searchline = tk.Frame(native, bg=SURFACE)
        searchline.pack(fill="x")
        self.hf_search_var = tk.StringVar()
        self._entry(searchline, self.hf_search_var).pack(side="left", fill="x", expand=True, ipady=6)
        self._button(searchline, "検索", self.search_huggingface).pack(side="left", padx=(8, 0))
        self.hf_results = tk.Listbox(native, height=3, bg=SURFACE_ALT,
                                     fg=FOREGROUND, selectbackground="#35594a",
                                     highlightthickness=0, borderwidth=0)
        self.hf_results.pack(fill="x", pady=(7, 5))
        self.hf_results.bind("<<ListboxSelect>>", self._select_hf_result)
        self._label(native, "Hugging Face リポジトリ（公開GGUF）",
                    color=MUTED, bg=SURFACE, size=9).pack(anchor="w", pady=(8, 2))
        repo_line = tk.Frame(native, bg=SURFACE)
        repo_line.pack(fill="x")
        self.hf_repo_var = tk.StringVar()
        self._entry(repo_line, self.hf_repo_var).pack(side="left", fill="x", expand=True, ipady=6)
        self._button(repo_line, "ファイル一覧", self.refresh_hf_files).pack(side="left", padx=(8, 0))
        self.hf_files = tk.Listbox(native, height=5, bg=SURFACE_ALT, fg=FOREGROUND,
                                   selectbackground="#35594a", highlightthickness=0,
                                   borderwidth=0, exportselection=False)
        self.hf_files.pack(fill="x", pady=(10, 8))
        self._button(native, "選択したGGUFをダウンロード", self.download_native,
                     strong=True).pack(fill="x")
        self._label(native, "新規取得にはInternet接続が必要です。",
                    color=MUTED, bg=SURFACE, size=8).pack(anchor="w", pady=(8, 0))

        delegated = self._section(page, "02  LM Studio / Ollama")
        delegated.grid(row=1, column=1, sticky="nsew", padx=(7, 0), pady=(0, 10))
        self.delegate_var = tk.StringVar(value="lmstudio")
        self._combobox(delegated, self.delegate_var, ["lmstudio", "ollama"]).pack(
            fill="x", pady=(0, 8))
        self.delegate_var.trace_add("write", lambda *_: self.refresh_delegated())
        self.delegate_id_var = tk.StringVar()
        self._entry(delegated, self.delegate_id_var).pack(fill="x", ipady=6)
        self._label(delegated, "既存のローカルサーバーへモデル取得を依頼",
                    color=MUTED, bg=SURFACE, size=8).pack(anchor="w", pady=(5, 9))
        self._button(delegated, "モデル取得を開始", self.download_delegate).pack(
            fill="x", pady=(0, 9))
        self.delegate_tree = ttk.Treeview(
            delegated, columns=("model", "size"), show="headings",
            height=5, style="Studio.Treeview", selectmode="browse")
        self.delegate_tree.heading("model", text="インストール済み")
        self.delegate_tree.heading("size", text="容量")
        self.delegate_tree.column("model", stretch=True, width=185)
        self.delegate_tree.column("size", stretch=False, width=85)
        self.delegate_tree.pack(fill="both", expand=True)
        delegate_buttons = tk.Frame(delegated, bg=SURFACE)
        delegate_buttons.pack(fill="x", pady=(9, 0))
        self._button(delegate_buttons, "一覧更新", self.refresh_delegated).pack(
            side="left", padx=(0, 8))
        self._button(delegate_buttons, "選択モデルをロード", self.load_delegated).pack(side="left")

        installed = self._section(page, "03  保存済みGGUF → ローカルで起動")
        installed.grid(row=2, column=0, sticky="nsew", padx=(0, 7))
        self.native_tree = ttk.Treeview(
            installed, columns=("file", "size"), show="headings",
            height=7, style="Studio.Treeview", selectmode="browse")
        self.native_tree.heading("file", text="検証済みGGUF")
        self.native_tree.heading("size", text="容量")
        self.native_tree.column("file", width=230)
        self.native_tree.column("size", width=82, stretch=False)
        self.native_tree.pack(fill="both", expand=True)
        self._label(installed, "コンテキスト長 / GPUレイヤー（空欄はランタイム既定値）",
                    color=MUTED, bg=SURFACE, size=9).pack(anchor="w", pady=(7, 4))
        controls = tk.Frame(installed, bg=SURFACE)
        controls.pack(fill="x")
        self.context_var = tk.IntVar(value=4096)
        self._entry(controls, self.context_var, width=8).pack(side="left", fill="x",
                                                               expand=True, ipady=5)
        self.gpu_layers_var = tk.StringVar(value="")
        self._entry(controls, self.gpu_layers_var, width=8).pack(side="left",
                                                                  padx=(9, 0), ipady=5)
        buttons = tk.Frame(installed, bg=SURFACE)
        buttons.pack(fill="x", pady=(9, 3))
        self._button(buttons, "GGUFを起動", self.launch_native, strong=True).pack(
            side="left", padx=(0, 7))
        self._button(buttons, "停止", self.stop_native).pack(side="left", padx=(0, 7))
        self._button(buttons, "削除", self.delete_native).pack(side="left", padx=(0, 7))
        self._button(buttons, "更新", self.refresh_native_list).pack(side="left")
        self.runtime_label = self._label(installed, "llama-serverの状態: 未確認",
                                         color=MUTED, bg=SURFACE, size=9)
        self.runtime_label.pack(anchor="w", pady=(9, 0))

        downloads = self._section(page, "04  ダウンロード進捗")
        downloads.grid(row=2, column=1, sticky="nsew", padx=(7, 0))
        self.jobs_tree = ttk.Treeview(
            downloads, columns=("model", "state", "progress"),
            show="headings", style="Studio.Treeview", height=8)
        for key, name, width in (
            ("model", "モデル", 180), ("state", "状態", 80), ("progress", "進捗", 105)
        ):
            self.jobs_tree.heading(key, text=name)
            self.jobs_tree.column(key, width=width, stretch=(key == "model"))
        self.jobs_tree.pack(fill="both", expand=True)
        self._button(downloads, "選択したironasuダウンロードを中断",
                     self.cancel_download).pack(anchor="w", pady=(9, 0))
        self.library_status = self._label(
            downloads, "モデルを検索またはローカル一覧から選択してください。",
            color=MUTED, bg=SURFACE, size=8, wraplength=330)
        self.library_status.pack(anchor="w", pady=(9, 0))


    def _build_agents(self) -> None:
        """Native controls for a read-only, bounded map/reduce corpus team."""
        tk, ttk = self.tk, self.ttk
        page = self.agent_tab
        page.grid_columnconfigure(0, weight=1)
        page.grid_rowconfigure(2, weight=1)
        header = tk.Frame(page, bg=BACKGROUND, pady=12)
        header.grid(row=0, column=0, sticky="ew")
        self._label(header, "コーパス並列エージェント", size=15, bold=True).pack(side="left")
        self._label(header, "最大16ワーカー · モデル同時処理は独立制限",
                    color=MUTED, size=9).pack(side="right")
        options = self._section(page, "01  タスクとコーパスの設定")
        options.grid(row=1, column=0, sticky="ew", pady=(0, 9))
        options.grid_columnconfigure(0, weight=1)
        self._label(options, "分析指示（全エージェント共通）", color=MUTED,
                    bg=SURFACE, size=9).grid(row=0, column=0, sticky="w")
        self.agent_objective = tk.Text(
            options, height=2, wrap="word", fg=FOREGROUND, bg=SURFACE_ALT,
            insertbackground=MINT, relief="flat", padx=8, pady=6)
        self.agent_objective.insert("1.0", "コーパスの要点・相違点・根拠を整理してください。")
        self.agent_objective.grid(row=1, column=0, columnspan=7, sticky="ew", pady=(5, 11))
        self._label(options, "コーパス本文（貼り付け可能）", color=MUTED,
                    bg=SURFACE, size=9).grid(row=2, column=0, sticky="w")
        self._button(options, "テキストファイルを読み込む", self.open_corpus_file).grid(
            row=2, column=1, columnspan=3, sticky="e", padx=(8, 0))
        self.corpus_input = tk.Text(
            options, height=6, wrap="word", fg=FOREGROUND,
            bg=SURFACE_ALT, insertbackground=MINT,
            relief="flat", padx=8, pady=6)
        self.corpus_input.grid(row=3, column=0, columnspan=7, sticky="ew", pady=(6, 7))
        self.corpus_note = self._label(
            options, "UTF-8テキスト: 最大100万文字。本文を削除するとファイル選択を解除します。",
            bg=SURFACE, color=MUTED, size=8)
        self.corpus_note.grid(row=4, column=0, columnspan=7, sticky="w", pady=(0, 10))
        self.corpus_input.bind("<<Modified>>", self._corpus_edited)
        self.corpus_input.edit_modified(False)
        self._label(options, "分割数", color=MUTED, bg=SURFACE, size=9).grid(
            row=5, column=0, sticky="w")
        self._label(options, "エージェント数", color=MUTED, bg=SURFACE, size=9).grid(
            row=5, column=2, sticky="w", padx=(12, 0))
        self._label(options, "モデル同時呼び出し", color=MUTED, bg=SURFACE, size=9).grid(
            row=5, column=4, sticky="w", padx=(12, 0))
        self.partitions_var = tk.StringVar(value="4")
        self.workers_var = tk.StringVar(value="4")
        self.model_slots_var = tk.StringVar(value="2")
        self._combobox(options, self.partitions_var, ["4", "16"], width=7).grid(
            row=6, column=0, sticky="w", pady=(4, 5))
        self._combobox(options, self.workers_var,
                       [str(i) for i in (1, 2, 4, 8, 12, 16)], width=7).grid(
            row=6, column=2, sticky="w", padx=(12, 0), pady=(4, 5))
        self._combobox(options, self.model_slots_var,
                       [str(i) for i in (1, 2, 4, 8, 16)], width=7).grid(
            row=6, column=4, sticky="w", padx=(12, 0), pady=(4, 5))
        action_row = tk.Frame(options, bg=SURFACE)
        action_row.grid(row=7, column=0, columnspan=7, sticky="ew", pady=(8, 0))
        self.agent_go = self._button(action_row, "並列分析を開始", self.start_agents,
                                     strong=True)
        self.agent_go.pack(side="left", padx=(0, 7))
        self.agent_stop = self._button(action_row, "■ 中断", self.stop_agents)
        self.agent_stop.pack(side="left")
        self.agent_stop.config(state="disabled")
        self.agent_progress = ttk.Progressbar(action_row, orient="horizontal",
                                               length=180, maximum=100,
                                               style="Studio.TProgressbar")
        self.agent_progress.pack(side="right", padx=8)
        self.agent_state = self._label(options, "未実行", bg=SURFACE, color=MUTED, size=9)
        self.agent_state.grid(row=8, column=0, columnspan=7, sticky="w", pady=(9, 0))
        result_area = self._section(page, "02  エージェント結果・統合レポート")
        result_area.grid(row=2, column=0, sticky="nsew")
        self.agent_output = self.ScrolledText(
            result_area, wrap="word", bg=SURFACE_ALT, fg=FOREGROUND, height=10,
            insertbackground=MINT, borderwidth=0, relief="flat", padx=12,
            pady=12, font=("Segoe UI", 10), state="disabled")
        self.agent_output.pack(fill="both", expand=True)
        self._button(result_area, "レポートをJSONで保存", self.export_agents).pack(
            anchor="e", pady=(9, 0))

    def _corpus_edited(self, _event: Any = None) -> None:
        if self.corpus_input.edit_modified():
            self.corpus_file_text = None
            self.corpus_file_path = ""
            self.corpus_input.edit_modified(False)
            self.corpus_note.config(text="本文を編集しました。入力欄のテキストが分析対象です。")

    def open_corpus_file(self) -> None:
        selected = self.filedialog.askopenfilename(
            title="分析するUTF-8テキストを選択",
            filetypes=[("Text & data", "*.txt *.md *.csv *.json *.log"),
                       ("All files", "*.*")])
        if not selected:
            return
        try:
            file = Path(selected)
            if not file.is_file() or file.stat().st_size > 4 * MAX_CORPUS_CHARS:
                raise ValueError("ファイルが大きすぎるか、ファイルではありません。")
            data = file.read_text(encoding="utf-8-sig")
            if len(data) > MAX_CORPUS_CHARS:
                raise ValueError("コーパスは100万文字までです。")
        except (OSError, UnicodeError, ValueError) as exc:
            self._show_error(exc)
            return
        self.corpus_file_text = data
        self.corpus_file_path = selected
        self.corpus_input.delete("1.0", "end")
        self.corpus_input.insert("1.0", data[:6000])
        self.corpus_input.edit_modified(False)
        self.corpus_note.config(
            text=f"ファイルから {len(data):,}文字を読み込み（入力欄は冒頭6000文字のプレビュー）")

    def _agent_corpus(self) -> str:
        return (self.corpus_file_text if self.corpus_file_text is not None
                else self.corpus_input.get("1.0", "end-1c"))

    def start_agents(self) -> None:
        if self.agents_running:
            return
        try:
            corpus = self._agent_corpus()
            objective = self.agent_objective.get("1.0", "end-1c").strip()
            partitions = int(self.partitions_var.get())
            workers = int(self.workers_var.get())
            slots = int(self.model_slots_var.get())
            mode, direct = normalized_model_selection(self.model_var.get())
            model = direct or mode
            if not corpus.strip():
                raise ValueError("コーパスを入力するかテキストファイルを選択してください。")
            if not 1 <= workers <= 16 or partitions not in (4, 16):
                raise ValueError("分割数は4/16、エージェント数は最大16です。")
            effective_slots = min(slots, self.config.max_concurrency)
            if not 1 <= effective_slots <= 16:
                raise ValueError("モデル呼び出し制限が不正です。")
            self.agent_total_segments = sum(len(split_segments(part))
                                            for part in split_corpus(corpus, partitions))
        except (ValueError, self.tk.TclError) as exc:
            self._show_error(exc)
            return
        self.agent_cancel = threading.Event()
        cancel = self.agent_cancel
        self.agents_running = True
        self.agent_go.configure(state="disabled")
        self.agent_stop.configure(state="normal")
        self.agent_progress.configure(value=0)
        self.agent_state.config(text=(
            f"{partitions}分割 × {workers}エージェント、モデル同時{effective_slots}枠 "
            f"(設定上限: {self.config.max_concurrency})"))
        self.agent_output.config(state="normal")
        self.agent_output.delete("1.0", "end")
        self.agent_output.insert("end", "分析中。コーパスを各エージェントに割り振っています。\n")
        self.agent_output.config(state="disabled")
        self.agent_report = None
        agent_engine = self.engine

        def job() -> Any:
            team = CorpusAgentTeam(agent_engine)
            return team.run(
                corpus, objective, partitions=partitions, workers=workers,
                model_slots=effective_slots, model=model, cancel=cancel,
                progress=lambda part, local_done, local_total, done: self.tasks.put((
                    "agents_progress", partitions, part, local_done,
                    local_total, done)),
            )
        self._async(job, self._agents_done, self._agents_error)

    def _agents_error(self, error: Exception) -> None:
        self.agents_running = False
        self.agent_go.config(state="normal")
        self.agent_stop.config(state="disabled")
        self._show_error(error)
        self.agent_state.config(text="分析失敗: " + str(error)[:180])

    def _agents_done(self, report: Any) -> None:
        self.agents_running = False
        self.agent_go.config(state="normal")
        self.agent_stop.config(state="disabled")
        self.agent_report = report
        self.agent_progress.configure(value=100 if report.complete else self.agent_progress["value"])
        self.agent_state.config(
            text=("完了" if report.complete else "不完全 / 一部失敗または中断")
            + f" · {report.corpus_chars:,}文字 / {report.partitions}分割 / "
              f"{report.total_segments}セグメント")
        self.agent_output.config(state="normal")
        self.agent_output.delete("1.0", "end")
        self.agent_output.insert("end", "【統合結果】\n" + (
            report.synthesis or "全パートの分析が成功していないため完全な統合結果はありません")
            + "\n\n")
        for part in report.parts:
            self.agent_output.insert(
                "end", f"【エージェント {part.index + 1:02d}】"
                f" {part.status} · {part.segment_count}セグメント\n")
            if part.error:
                self.agent_output.insert("end", "エラー: " + part.error + "\n")
            self.agent_output.insert("end", (part.summary or "—") + "\n\n")
        self.agent_output.config(state="disabled")

    def stop_agents(self) -> None:
        if self.agents_running:
            self.agent_cancel.set()
            self.agent_state.config(text="停止要求を受付。進行中のモデル呼び出しは完了待ちです。")
            self.agent_stop.config(state="disabled")

    def export_agents(self) -> None:
        if self.agent_report is None:
            self._show_error("先にエージェント分析を実行してください。")
            return
        selected = self.filedialog.asksaveasfilename(
            title="エージェントレポートをJSON保存",
            defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not selected:
            return
        report = self.agent_report
        data = {
            "complete": report.complete, "partitions": report.partitions,
            "workers": report.workers, "model_slots": report.model_slots,
            "corpus_chars": report.corpus_chars,
            "total_segments": report.total_segments,
            "synthesis": report.synthesis,
            "parts": [part.__dict__ for part in report.parts],
        }
        try:
            Path(selected).write_text(json.dumps(
                data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            self._show_error(exc)

    @staticmethod
    def _bytes(size: Any) -> str:
        if type(size) is not int or size < 0:
            return "—"
        num = float(size)
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if num < 1024.0 or unit == "TiB":
                return f"{num:.1f} {unit}"
            num /= 1024
        return "—"

    def _async(self, work: Callable[[], Any], on_done: Callable[[Any], None],
               on_error: Callable[[Exception], None] | None = None) -> None:
        def worker() -> None:
            try:
                result = work()
                self.tasks.put(("task", on_done, result, None))
            except Exception as exc:
                self.tasks.put(("task", on_error, None, exc))
        threading.Thread(target=worker, daemon=True, name="ironasu-desktop-task").start()

    def _drain(self) -> None:
        if self.closed:
            return
        handled = 0
        while handled < 120:
            try:
                item = self.tasks.get_nowait()
            except queue.Empty:
                break
            handled += 1
            kind = item[0]
            if kind == "task":
                _, fn, result, error = item
                if error is not None:
                    if fn is None:
                        self._show_error(error)
                    else:
                        fn(error)
                elif fn is not None:
                    fn(result)
            elif kind == "delta":
                _, sid, msg_index, segment, text = item
                if sid < len(self.sessions) and msg_index < len(self.sessions[sid].messages):
                    msg = self.sessions[sid].messages[msg_index]
                    if segment == "reasoning":
                        msg["reasoning"] = msg.get("reasoning", "") + text
                    else:
                        msg["content"] = msg.get("content", "") + text
                    if self.session_index == sid:
                        self._render_conversation()
            elif kind == "generation_end":
                _, sid, msg_index, status, stats, failure = item
                if sid < len(self.sessions) and msg_index < len(self.sessions[sid].messages):
                    msg = self.sessions[sid].messages[msg_index]
                    msg["status"] = status
                    if status != "completed":
                        msg["failed"] = True
                    if self.session_index == sid:
                        self._render_conversation()
                self.busy = False
                self.send_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.provider_combo.configure(state="readonly")
                if stats:
                    def fmt(v: Any, unit: str) -> str:
                        return "—" if v is None else f"{v:.2f}{unit}"
                    self.metrics_label.config(
                        text="TTFT: " + fmt(stats["ttft_s"], " s") +
                        "\n生成速度: " + fmt(stats["tokens_per_s"], " tok/s") +
                        "\nReasoning受信: " + fmt(stats["reasoning_s"], " s") +
                        "\n文字/秒: " + fmt(stats["chars_per_s"], " char/s"))
                if failure:
                    self._show_error(failure)
            elif kind == "agents_progress":
                _, parts, agent, local_done, local_total, completed = item
                # Segment count isn't uniformly distributed; this progress
                # indicator counts completed subsegments, not GPU FLOPs.
                self.agent_state.config(
                    text=f"分析中: エージェント {agent + 1}/{parts} "
                         f"パート内 {local_done}/{local_total} セグメント完了")
                if self.agent_total_segments > 0:
                    self.agent_progress.config(
                        value=min(99, int(100 * completed / self.agent_total_segments)))
            elif kind == "status":
                self.composer_note.config(text=item[1])
        self.root.after(80, self._drain)

    def _tick(self) -> None:
        if self.closed:
            return
        snapshot = self.engine.diagnostic_snapshot()
        self.load_note.config(text="稼働中: " + str(snapshot.get("active_requests", 0)) +
                              "  /  待機中: " + str(snapshot.get("waiting_requests", 0)))
        state = self.managed.status()
        summary = state["state"] + " · " + (state.get("detail") or "")
        if summary != self.last_status:
            self.last_status = summary
            self.runtime_label.config(text="管理ランタイム: " + summary[:105])
        if state.get("model_id") != self.last_model_id:
            self.last_model_id = state.get("model_id")
            self.refresh_connection()
        if time.monotonic() - self._last_library_refresh > 1.5:
            self._last_library_refresh = time.monotonic()
            self._render_jobs()
        self.root.after(1500, self._tick)

    def _show_error(self, error: Any) -> None:
        self.composer_note.config(text="エラー: " + str(error)[:160], fg=WARN)
        if hasattr(self, "library_status"):
            self.library_status.config(text=str(error)[:220], fg=WARN)

    def _render_sessions(self) -> None:
        self.session_list.delete(0, "end")
        for s in self.sessions:
            self.session_list.insert("end", "▤  " + s.title[:27])
        self.session_list.selection_clear(0, "end")
        self.session_list.selection_set(self.session_index)

    def _render_conversation(self) -> None:
        session = self.sessions[self.session_index]
        self.chat_title.config(text=session.title[:65])
        field = self.transcript
        y = field.yview()
        at_bottom = y[1] > 0.93
        field.configure(state="normal")
        field.delete("1.0", "end")
        if not session.messages:
            field.insert("end", "\n\n  ironasu Studio\n", "assistant_title")
            field.insert("end", "\n  独立したデスクトップ画面でローカルLLMを操作できます。\n"
                         "  モデルの接続先を選び、下の入力欄からチャットを始めてください。\n", "body")
        for m in session.messages:
            user = m.get("role") == "user"
            field.insert("end", "\nあなた\n" if user else "\nironasu\n",
                         "user_title" if user else "assistant_title")
            if not user and m.get("reasoning") and self.show_reasoning.get():
                field.insert("end", "  Reasoning（モデルの出力）\n", "thinking_title")
                field.insert("end", m.get("reasoning", "") + "\n", "thinking")
            body = m.get("content", "")
            if not body and not user and not m.get("failed"):
                body = "生成中…"
            field.insert("end", body + "\n", "body")
            if not user and m.get("status"):
                field.insert("end", "  " + str(m["status"]) + "\n", "status")
        field.configure(state="disabled")
        if at_bottom or len(session.messages) < 5:
            field.see("end")
        else:
            field.yview_moveto(y[0])

    def new_session(self) -> None:
        if self.busy:
            self._show_error("生成中はセッションを切り替えられません。")
            return
        self.sessions.insert(0, Session())
        self.session_index = 0
        self._render_sessions()
        self._render_conversation()

    def _choose_session(self, _event: Any) -> None:
        indices = self.session_list.curselection()
        if indices and not self.busy:
            self.session_index = int(indices[0])
            self._render_conversation()

    def delete_session(self) -> None:
        if self.busy:
            return
        if not self.messagebox.askyesno("確認", "現在のチャットを削除しますか？"):
            return
        self.sessions.pop(self.session_index)
        if not self.sessions:
            self.sessions.append(Session())
        self.session_index = 0
        self._render_sessions()
        self._render_conversation()

    def export_session(self) -> None:
        selected = self.filedialog.asksaveasfilename(
            title="会話をJSON保存", defaultextension=".json",
            filetypes=[("JSON", "*.json")])
        if not selected:
            return
        session = self.sessions[self.session_index]
        data = {"title": session.title, "messages": [
            {k: v for k, v in msg.items() if k in ("role", "content", "reasoning", "status")}
            for msg in session.messages]}
        try:
            Path(selected).write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            self._show_error(exc)

    def _change_provider(self, _event: Any = None) -> None:
        if self.busy:
            return
        selection = self.provider_var.get()
        self.provider = selection
        # Custom endpoint from CLI keeps its explicit URL; user can return to
        # auto or pick an explicit local preset via the native dropdown.
        if selection == "auto":
            backend = AutoHTTPBackend(self.config)
        else:
            backend = HTTPBackend(replace(self.config, backend_url=PRESETS[selection]))
        self.upstream = backend
        self.mux = RuntimeBackendMux(backend, self.managed)
        self.engine = AdaptiveEngine(self.config, self.mux)
        self.refresh_connection()

    def refresh_connection(self) -> None:
        if self.closed:
            return
        self._last_connection_refresh = time.monotonic()
        def fetch() -> tuple[list[str] | None, dict[str, Any]]:
            return self.mux.list_models(), self.mux.endpoint()
        def on_done(value: tuple[list[str] | None, dict[str, Any]]) -> None:
            names, endpoint = value
            old = self.model_var.get()
            self.available_models = names or []
            self.model_combo.configure(values=["auto", "fast", "quality", *self.available_models])
            if old not in ("auto", "fast", "quality", *self.available_models):
                self.model_var.set("auto")
            provider = endpoint.get("provider") or self.provider
            self.status_label.config(
                text=(provider + (" · モデル " + str(len(names)) if names is not None
                                  else " · サーバー未検出"))[:70],
                fg=MINT if names is not None else WARN)
        self._async(fetch, on_done)

    def _messages_payload(self, sid: int) -> dict[str, Any]:
        session = self.sessions[sid]
        messages: list[dict[str, str]] = []
        system = self.system_prompt.get("1.0", "end-1c").strip()
        if system:
            messages.append({"role": "system", "content": system})
        for m in session.messages:
            if m.get("failed") or not m.get("content"):
                continue
            if m["role"] in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})
        mode, selected = normalized_model_selection(self.model_var.get())
        if not 1 <= self.tokens_var.get() <= 32768:
            raise ValueError("最大生成トークンは1〜32768です")
        payload: dict[str, Any] = {
            "model": "ironasu", "messages": messages, "stream": self.stream_enabled.get(),
            "temperature": float(self.temperature_var.get()),
            "top_p": float(self.top_p_var.get()), "max_tokens": self.tokens_var.get(),
            "x_ironasu_mode": mode,
            "x_ironasu_cache": True,
        }
        if selected:
            payload["x_ironasu_model"] = selected
        return payload

    def _send_key(self, _event: Any) -> str:
        self.send()
        return "break"

    def send(self) -> None:
        if self.busy:
            return
        prompt = self.prompt.get("1.0", "end-1c").strip()
        if not prompt:
            return
        sid = self.session_index
        session = self.sessions[sid]
        if session.title == "新しいチャット":
            session.title = " ".join(prompt.split())[:45]
        session.messages.append({"role": "user", "content": prompt})
        session.messages.append({"role": "assistant", "content": "", "reasoning": ""})
        message_index = len(session.messages) - 1
        self.prompt.delete("1.0", "end")
        try:
            payload = self._messages_payload(sid)
        except (ValueError, self.tk.TclError) as exc:
            session.messages.pop()
            session.messages.pop()
            self._show_error(exc)
            return
        self.cancel_event = threading.Event()
        cancel = self.cancel_event
        self.busy = True
        self.send_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.provider_combo.configure(state="disabled")
        self._render_sessions()
        self._render_conversation()
        autoload = (self.autoload_var.get() and
                    self.provider_var.get() in ("lmstudio", "auto") and
                    isinstance(payload.get("x_ironasu_model"), str) and
                    not payload["x_ironasu_model"].startswith("ironasu-local-"))
        threading.Thread(
            target=self._generation_worker,
            args=(sid, message_index, payload, autoload, cancel),
            daemon=True, name="ironasu-native-chat",
        ).start()

    def _generation_worker(self, sid: int, index: int, payload: dict[str, Any],
                           autoload: bool, cancel: threading.Event) -> None:
        stats = GenerationStats()
        done = "completed"
        failure: Exception | None = None
        try:
            if autoload:
                endpoint = self.mux.endpoint().get("url")
                if not isinstance(endpoint, str) or not endpoint.startswith(
                        "http://127.0.0.1:1234/"):
                    raise InferenceError(
                        "自動ロードは1234番のローカルLM Studio接続時のみ許可", 409)
                self.model_manager.ensure_loaded("lmstudio", payload["x_ironasu_model"])
            if cancel.is_set():
                raise InterruptedError("Stopped")
            if payload["stream"]:
                with self.engine.stream(payload) as response:
                    with self.stream_lock:
                        self.active_stream = response
                    decoder = SSEEvents()
                    for line in response:
                        if cancel.is_set():
                            raise InterruptedError("Stopped")
                        for chunk in decoder.feed(line):
                            updates = stats.ingest(chunk, time.perf_counter())
                            for typ, text in updates:
                                self.tasks.put(("delta", sid, index, typ, text))
                    tail = stats.append_content("", time.perf_counter(), flush=True)
                    for typ, text in tail:
                        self.tasks.put(("delta", sid, index, typ, text))
            else:
                data = self.engine.complete(payload)
                choice = (data.get("choices") or [{}])[0]
                chunk = choice.get("message") or {}
                reason = extract_reasoning(chunk)
                if reason:
                    stats.reasoning = reason
                    self.tasks.put(("delta", sid, index, "reasoning", reason))
                parts = stats.append_content(extract_text(chunk), time.perf_counter(), flush=True)
                for typ, text in parts:
                    self.tasks.put(("delta", sid, index, typ, text))
                usage = data.get("usage")
                if isinstance(usage, dict):
                    value = usage.get("completion_tokens")
                    if type(value) is int:
                        stats.tokens = value
                # No TTFT claims can be made from a non-streamed response.
                stats.first = None
                stats.first_reasoning = None
                stats.first_answer = None
            if cancel.is_set():
                raise InterruptedError("Stopped")
        except (InterruptedError, OSError) as exc:
            if cancel.is_set():
                done = "stopped"
            else:
                done, failure = "failed", exc
        except Exception as exc:
            done, failure = "failed", exc
        finally:
            with self.stream_lock:
                self.active_stream = None
            metrics = stats.finish(time.perf_counter())
            self.tasks.put(("generation_end", sid, index, done, metrics, failure))

    def stop_generation(self) -> None:
        self.cancel_event.set()
        self.agent_cancel.set()
        with self.stream_lock:
            response = self.active_stream
        if response is not None:
            try:
                response.close()
            except OSError:
                pass

    def search_huggingface(self) -> None:
        query = self.hf_search_var.get().strip()
        self.library_status.config(text="Hugging Faceカタログを検索中…", fg=MUTED)
        def done(rows: list[dict[str, Any]]) -> None:
            self.native_results = rows
            self.hf_results.delete(0, "end")
            for row in rows:
                self.hf_results.insert("end", row["id"])
            self.library_status.config(
                text=f"{len(rows)}件の候補。検索結果を選択してファイル一覧を開いてください。",
                fg=MUTED)
        self._async(lambda: self.native.search(query), done)

    def _select_hf_result(self, _event: Any) -> None:
        index = self.hf_results.curselection()
        if index and int(index[0]) < len(self.native_results):
            self.hf_repo_var.set(self.native_results[int(index[0])]["id"])

    def refresh_hf_files(self) -> None:
        repo = self.hf_repo_var.get().strip()
        self.library_status.config(text="公開GGUFのファイル情報を取得中…", fg=MUTED)
        def done(files: list[dict[str, Any]]) -> None:
            self.native_files = files
            self.hf_files.delete(0, "end")
            for item in files:
                self.hf_files.insert(
                    "end", item["path"] + " · " + self._bytes(item.get("size_bytes")))
            self.library_status.config(
                text=f"{len(files)}件。ライセンス・容量・必要VRAMを確認してください。",
                fg=MUTED)
        self._async(lambda: self.native.files(repo), done)

    def download_native(self) -> None:
        selected = self.hf_files.curselection()
        if not selected or int(selected[0]) >= len(self.native_files):
            self._show_error("先にGGUFファイルを選んでください。")
            return
        item = self.native_files[int(selected[0])]
        repo = self.hf_repo_var.get().strip()
        if not self.messagebox.askyesno(
            "モデルのダウンロード",
            f"{item['path']}\n{self._bytes(item.get('size_bytes'))}\n"
            "このGGUFをHugging Faceから保存しますか？"):
            return
        self._async(lambda: self.native.start(repo, item["path"]),
                    lambda _job: self._render_jobs())

    def refresh_native_list(self) -> None:
        def done(rows: list[dict[str, Any]]) -> None:
            self.native_installed = rows
            selected = self.native_tree.selection()
            old = self.native_tree.item(selected[0], "values")[0] if selected else None
            self.native_tree.delete(*self.native_tree.get_children())
            for i, item in enumerate(rows):
                self.native_tree.insert("", "end", iid=str(i),
                                        values=(item["label"], self._bytes(item["size_bytes"])))
            if old:
                for i, item in enumerate(rows):
                    if item["label"] == old:
                        self.native_tree.selection_set(str(i))
                        break
        self._async(self.native.installed, done)

    def _native_selected(self) -> dict[str, Any] | None:
        chosen = self.native_tree.selection()
        if chosen and int(chosen[0]) < len(self.native_installed):
            return self.native_installed[int(chosen[0])]
        self._show_error("保存済みGGUFを一覧から選択してください。")
        return None

    def launch_native(self) -> None:
        item = self._native_selected()
        if item is None:
            return
        try:
            context = self.context_var.get()
            gpu_raw = self.gpu_layers_var.get().strip()
            gpu = int(gpu_raw) if gpu_raw else None
        except (ValueError, self.tk.TclError) as exc:
            self._show_error(exc)
            return
        self._async(lambda: self.managed.start(item["id"], context_length=context,
                                               gpu_layers=gpu),
                    lambda _status: self.runtime_label.config(
                        text="GGUFの署名を検証し、ローカル実行エンジンを起動しています…"))

    def stop_native(self) -> None:
        self._async(self.managed.stop, lambda _status: self.refresh_connection())

    def delete_native(self) -> None:
        item = self._native_selected()
        if item is None:
            return
        state = self.managed.status()
        if state.get("installed_id") == item["id"] and state.get("state") in (
                "starting", "running"):
            self._show_error("モデルを停止してから削除してください。")
            return
        if not self.messagebox.askyesno("削除の確認",
                                        item["label"] + "\nローカルGGUFファイルを削除しますか？"):
            return
        self._async(lambda: self.native.remove(item["id"]),
                    lambda _result: self.refresh_native_list())

    def download_delegate(self) -> None:
        provider = self.delegate_var.get()
        model_id = self.delegate_id_var.get().strip()
        if not model_id:
            self._show_error("モデルIDを入力してください。")
            return
        if not self.messagebox.askyesno(
            "ダウンロードの確認", f"{provider} に {model_id} の取得を依頼しますか？"):
            return
        self._async(lambda: self.model_manager.start(provider, model_id),
                    lambda _job: self._render_jobs())

    def refresh_delegated(self) -> None:
        provider = self.delegate_var.get()
        def done(rows: list[dict[str, Any]]) -> None:
            if self.delegate_var.get() != provider:
                return
            self.delegated_installed = rows
            self.delegate_tree.delete(*self.delegate_tree.get_children())
            for i, item in enumerate(rows):
                self.delegate_tree.insert("", "end", iid=str(i),
                                          values=(item["id"], self._bytes(item.get("size_bytes"))))
        self._async(lambda: self.model_manager.installed(provider), done)

    def load_delegated(self) -> None:
        chosen = self.delegate_tree.selection()
        if not chosen or int(chosen[0]) >= len(self.delegated_installed):
            self._show_error("ロードするモデルを選択してください。")
            return
        if self.delegate_var.get() != "lmstudio":
            self._show_error("手動ロードはLM Studioモデルのみ対応しています。")
            return
        model_id = self.delegated_installed[int(chosen[0])]["id"]
        self._async(lambda: self.model_manager.load("lmstudio", model_id),
                    lambda _result: (self.refresh_delegated(), self.refresh_connection()))

    def _render_jobs(self) -> None:
        jobs = self.native.jobs() + self.model_manager.list_jobs()
        signature = repr([(j["id"], j["state"], j.get("completed_bytes")) for j in jobs])
        if signature == self._jobs_cache:
            return
        self._jobs_cache = signature
        self.jobs_tree.delete(*self.jobs_tree.get_children())
        for job in jobs[-30:]:
            total, done = job.get("total_bytes"), job.get("completed_bytes")
            value = (f"{100*done/total:.1f}%" if type(total) is int and total > 0
                     and type(done) is int else "—")
            self.jobs_tree.insert("", "end", iid=job["id"],
                                  values=(str(job["model"])[:50], job["state"], value))
        if any(job["state"] == "completed" for job in jobs):
            self.refresh_native_list()

    def cancel_download(self) -> None:
        selected = self.jobs_tree.selection()
        if not selected:
            return
        job_id = selected[0]
        if not job_id.startswith("native_"):
            self._show_error("この操作はironasu管理のダウンロードのみ対象です。")
            return
        self._async(lambda: self.native.cancel(job_id),
                    lambda _result: self._render_jobs())

    def refresh_library(self) -> None:
        self.refresh_native_list()
        self.refresh_delegated()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.cancel_event.set()
        with self.stream_lock:
            stream = self.active_stream
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
        self.managed.stop()
        self.root.destroy()


def main(argv: list[str] | None = None) -> int:
    args = desktop_parser().parse_args(argv)
    if args.self_test:
        desktop_self_test(args.self_test)
        return 0
    if args.smoke_gui:
        # Run the real widget constructor without opening a browser or
        # contacting a model server. Suitable for frozen Windows CI.
        import tkinter as tk
        class FakeBackend:
            def list_models(self):
                return ["local-test-model"]
            def endpoint(self):
                return {"provider": "fixture", "url": "http://127.0.0.1:1234/v1"}
            def complete(self, _payload):
                return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        root = tk.Tk()
        root.withdraw()
        window = DesktopWindow(root, Config(auto_models=True, fast_model="",
                                             quality_model=""), FakeBackend())
        root.update_idletasks()
        info = {
            "native_window_created": True,
            "tabs": [window.tabs.tab(tab, "text").strip() for tab in window.tabs.tabs()],
            "agents_max": 16,
            "corpus_partitions": [4, 16],
            "no_browser": True,
        }
        Path(args.smoke_gui).write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        window.close()
        return 0
    try:
        config, backend = configure(args)
    except ValueError as exc:
        desktop_parser().error(str(exc))
    # No web server or external browser is started in this entry point.
    import tkinter as tk
    root = tk.Tk()
    DesktopWindow(root, config, backend, provider=args.provider)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
