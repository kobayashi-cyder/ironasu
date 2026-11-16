"""ironasu Studio — independent Qt 6 desktop application.

No browser engine, embedded HTTP server, CDN or Tkinter is started here.
Qt Widgets are high-DPI aware; model inference and filesystem operations run
outside the GUI thread, and only the main Qt thread mutates widgets.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable
from uuid import uuid4
import argparse
import json
import os
import time
from urllib.parse import urlsplit

from .compat import AutoHTTPBackend, PRESETS
from .core import AdaptiveEngine, Config, HTTPBackend, InferenceError
from .desktop_logic import GenerationStats, SSEEvents, extract_reasoning, extract_text, normalized_model_selection
from .corpus_agents import CorpusAgentTeam, MAX_CORPUS_CHARS, split_corpus, split_segments
from .mission import MissionController, MissionJournal
from .models import ModelManager
from .native_models import NativeModelManager
from .runtime import ManagedGGUFRuntime, RuntimeBackendMux
from .studio import configure, parser as studio_parser

VERSION = "1.8.0"

# Kept import-light: --help and --self-test can be invoked on a system without
# Qt installed. Qt is imported inside build_ui()/main() only.
def parser() -> argparse.ArgumentParser:
    p = studio_parser()
    p.prog = "ironasu-studio"
    p.description = "Modern native ironasu Studio UI (PySide6/Qt6, no browser)"
    p.add_argument("--self-test", metavar="FILE", help="Write a desktop packaging diagnostic JSON")
    p.add_argument("--smoke-gui", metavar="FILE", help="Construct Qt Widgets and write diagnostics")
    return p


def self_test(path: str) -> None:
    from importlib.util import find_spec
    Path(path).write_text(json.dumps({
        "mode": "native-qt6-desktop", "version": VERSION,
        "qt_available": find_spec("PySide6") is not None,
        "browser_required": False, "http_server_required": False,
        "agents_max": 16, "corpus_partitions": [4, 16],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class Session:
    title: str = "新しいチャット"
    messages: list[dict[str, Any]] = field(default_factory=list)


THEME = r"""
QWidget { background: #10161f; color: #e8f2f1; font-size: 10pt; }
QLabel, QCheckBox { background: transparent; }
QMainWindow, QWidget#outer, QWidget#workArea { background: #0c121a; }
QFrame#sidebar { background: #121d27; border-right: 1px solid #263741; }
QFrame#inspector { background: #141f29; border-left: 1px solid #263741; }
QFrame#panel { background: #192631; border: 1px solid #30424b; border-radius: 14px; }
QFrame#bubbleUser { background: #233e39; border: 1px solid #3b5b50; border-radius: 15px; }
QFrame#bubbleAssistant { background: #1c2934; border: 1px solid #2b3a46; border-radius: 15px; }
QLabel#logo { color: #9de4c6; font-size: 20pt; font-weight: 800; }
QLabel#subtle, QLabel#caption { color: #9ab1bd; }
QLabel#title { color: #f0f8f7; font-size: 16pt; font-weight: 700; }
QLabel#section { color: #a0e8c9; font-size: 11pt; font-weight: 700; }
QLabel#metric { color: #edf8f7; font-size: 13pt; font-weight: 700; }
QLabel#role { color: #a0e8c9; font-weight: 700; }
QPushButton { background: #243744; color: #e6f4ef; border: 1px solid #3b515c;
    border-radius: 10px; padding: 9px 14px; min-height: 19px; }
QPushButton:hover { background: #304756; border-color: #739c94; }
QPushButton:pressed { background: #183e35; }
QPushButton:disabled { color: #6b8490; border-color: #2b3b44; background: #1e2b34; }
QPushButton#primary { background: #91ddbb; border-color: #91ddbb; color: #102820; font-weight: 700; }
QPushButton#primary:hover { background: #b0efd0; }
QPushButton#navActive { background: #253e37; color: #b2f6d2; border-color: #3b6858; text-align: left; }
QPushButton#navItem { background: transparent; border: none; text-align: left; }
QPushButton#navItem:hover { background: #22353d; }
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #101c27; color: #edf7f6; border: 1px solid #3b515d;
    border-radius: 9px; padding: 8px 11px; selection-background-color: #41715d; }
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #8ed9b9; }
QComboBox, QSpinBox, QDoubleSpinBox { min-height: 21px; }
/* Qt paints the native combobox arrow; do not erase it in QSS. */
QComboBox QAbstractItemView { background: #182631; color: #f1f9f6;
    border: 1px solid #516b72; selection-background-color: #335849; min-height: 30px; }
QTabWidget::pane { border: none; }
QTabBar::tab { background: transparent; color: #9db2c0; padding: 12px 19px;
    margin-right: 6px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #a5f1d0; border-bottom-color: #8fdfbe; font-weight: 700; }
QTabBar::tab:hover { color: #f0f6f4; }
QCheckBox { spacing: 9px; min-height: 27px; }
/* Keep Qt's native checked indicator so the tick remains visible on HiDPI displays. */
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #405461; border-radius: 5px; min-height: 32px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { height: 9px; background: transparent; }
QScrollBar::handle:horizontal { background: #405461; border-radius: 4px; }
QProgressBar { border: 1px solid #354e57; background: #15222c; border-radius: 7px;
    height: 13px; text-align: center; color: #e3f5ed; }
QProgressBar::chunk { border-radius: 6px; background: #87d9b5; }
QTreeWidget, QListWidget { border: 1px solid #2d434e; border-radius: 9px;
    background: #14222d; alternate-background-color: #192b36; outline: none; }
QTreeWidget::item, QListWidget::item { padding: 7px 6px; min-height: 26px; }
QTreeWidget::item:selected, QListWidget::item:selected { background: #305346; color: #e6ffef; }
QHeaderView::section { background: #202f39; color: #b6d7cb; border: none;
    border-bottom: 1px solid #304751; padding: 10px 9px; font-weight: 700; }
QSplitter::handle { background: #283a43; }
QSplitter::handle:hover { background: #5c9182; }
QToolButton { background: transparent; border: none; color: #95d8c0; text-align: left; padding: 6px; }
QToolButton:hover { background: #29463e; border-radius: 6px; }
QMessageBox { background: #172430; }
"""


def build_ui():
    """Define Qt subclasses lazily so core CLI tools remain dependency-light."""
    from PySide6.QtCore import Qt, QTimer, QObject, Signal, QSignalBlocker
    from PySide6.QtGui import QFont, QFontDatabase, QGuiApplication, QKeySequence, QShortcut
    from PySide6.QtWidgets import (
        QApplication, QWidget, QMainWindow, QFrame, QLabel, QPushButton, QVBoxLayout,
        QHBoxLayout, QGridLayout, QSplitter, QTabWidget, QScrollArea, QPlainTextEdit,
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QListWidget,
        QTreeWidget, QTreeWidgetItem, QHeaderView, QFileDialog, QMessageBox,
        QProgressBar, QToolButton, QSizePolicy,
    )

    def label(text="", *, kind=None, wrap=False):
        w = QLabel(text)
        if kind:
            w.setObjectName(kind)
        w.setWordWrap(wrap)
        w.setTextFormat(Qt.TextFormat.PlainText)
        w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return w

    def frame(name="panel"):
        w = QFrame()
        w.setObjectName(name)
        return w

    def button(text, callback=None, *, primary=False):
        w = QPushButton(text)
        if primary:
            w.setObjectName("primary")
        if callback is not None:
            w.clicked.connect(callback)
        return w

    def box(title):
        w = frame()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(17, 17, 17, 17)
        lay.setSpacing(11)
        lay.addWidget(label(title, kind="section"))
        return w, lay

    def scroller(child):
        w = QScrollArea()
        w.setWidgetResizable(True)
        w.setWidget(child)
        return w

    def row_layout(parent=None, margin=0, gap=9):
        lay = QHBoxLayout(parent)
        lay.setContentsMargins(margin, margin, margin, margin)
        lay.setSpacing(gap)
        return lay

    def col_layout(parent=None, margin=0, gap=9):
        lay = QVBoxLayout(parent)
        lay.setContentsMargins(margin, margin, margin, margin)
        lay.setSpacing(gap)
        return lay

    def tree(headers):
        w = QTreeWidget()
        w.setColumnCount(len(headers))
        w.setHeaderLabels(headers)
        w.setRootIsDecorated(False)
        w.setAlternatingRowColors(True)
        w.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        w.header().setStretchLastSection(False)
        w.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i in range(1, len(headers)):
            w.header().setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        w.setMinimumHeight(130)
        return w

    class Bridge(QObject):
        completed = Signal(str, object, object)
        stream_delta = Signal(int, int, str, str)
        generation_done = Signal(int, int, str, object, object)
        corpus_progress = Signal(int, int, int, int)
        mission_progress = Signal(object)

    class MessageCard(QFrame):
        def __init__(self, msg):
            super().__init__()
            self.setObjectName("bubbleUser" if msg["role"] == "user" else "bubbleAssistant")
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            layout = col_layout(self, 14, 10)
            heading = row_layout()
            self.title = label("あなた" if msg["role"] == "user" else "ironasu", kind="role")
            heading.addWidget(self.title)
            heading.addStretch()
            self.note = label(msg.get("status", ""), kind="caption")
            heading.addWidget(self.note)
            layout.addLayout(heading)
            self.reason_btn = QToolButton()
            self.reason_btn.setText("▸  Reasoning · モデル出力を表示")
            self.reason_btn.setCheckable(True)
            self.reason_btn.setChecked(False)
            self.reason = QPlainTextEdit()
            self.reason.setReadOnly(True)
            self.reason.setMaximumHeight(200)
            self.reason.setMinimumHeight(65)
            self.reason.hide()
            self.reason_btn.toggled.connect(self.reason.setVisible)
            layout.addWidget(self.reason_btn)
            layout.addWidget(self.reason)
            self.body = label("", wrap=True)
            self.body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
            self.body.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.body.setContentsMargins(1, 2, 1, 2)
            layout.addWidget(self.body)
            self.setMinimumWidth(250)
            self.update_message(msg, True)

        def update_message(self, msg, show_reason=True):
            self.note.setText(msg.get("status", ""))
            content = msg.get("content", "")
            self.body.setText(content if content else ("生成中…" if msg["role"] != "user" and not msg.get("failed") else ""))
            has_reason = bool(msg.get("reasoning")) and show_reason
            self.reason_btn.setVisible(has_reason)
            if has_reason:
                self.reason.setPlainText(msg["reasoning"])
            elif self.reason.isVisible():
                self.reason.hide()

    class MainWindow(QMainWindow):
        def __init__(self, config, backend, *, provider="auto", smoke=False):
            super().__init__()
            self.setWindowTitle("ironasu Studio  ·  Qt Desktop")
            self.resize(1510, 920)
            self.setMinimumSize(1080, 700)
            self.config = config
            self.provider = provider
            self.native = NativeModelManager()
            self.managed = ManagedGGUFRuntime(self.native)
            self.models = ModelManager(config)
            self.backend = backend
            self.mux = RuntimeBackendMux(backend, self.managed)
            self.engine = AdaptiveEngine(config, self.mux)
            self.sessions = [Session()]
            self.session_index = 0
            self.chat_busy = False
            self.agent_busy = False
            self.agent_cancel = Event()
            self.mission_busy = False
            self.mission_cancel = Event()
            self.mission_report = None
            self.chat_cancel = Event()
            self.stream_lock = Lock()
            self.active_stream = None
            self.pending = {}
            self.bridge = Bridge()
            self.bridge.completed.connect(self._task_complete)
            self.bridge.stream_delta.connect(self._stream_delta)
            self.bridge.generation_done.connect(self._generation_done)
            self.bridge.corpus_progress.connect(self._agent_progress)
            self.bridge.mission_progress.connect(self._mission_event)
            self.native_results = []
            self.native_files = []
            self.native_installed = []
            self.delegated_installed = []
            self.agent_report = None
            self.agent_total_segments = 0
            self.agent_file_text = None
            self.agent_file_path = ""
            self._agent_loading = False
            self._job_signature = ""
            self._last_runtime = ""
            self._last_runtime_model = None
            self._last_job_refresh = 0.0
            self._closed = False
            self._message_widgets = []
            self._construct()
            self._render_sessions()
            self._render_chat()
            self.poll_timer = QTimer(self)
            self.poll_timer.setInterval(1600)
            self.poll_timer.timeout.connect(self._poll)
            if not smoke:
                self.poll_timer.start()
                QTimer.singleShot(200, self.refresh_models)
                QTimer.singleShot(350, self.refresh_native)
                QTimer.singleShot(500, self.refresh_delegated)

        def _run(self, fn, on_ok=None, on_error=None):
            identity = uuid4().hex
            self.pending[identity] = (on_ok, on_error)
            def work():
                try:
                    value = fn()
                    self.bridge.completed.emit(identity, value, None)
                except Exception as e:
                    self.bridge.completed.emit(identity, None, e)
            Thread(target=work, daemon=True, name="ironasu-qt-task").start()

        def _task_complete(self, key, result, error):
            handlers = self.pending.pop(key, (None, None))
            if self._closed:
                return
            callback = handlers[1] if error is not None else handlers[0]
            if callback is not None:
                callback(error if error is not None else result)
            elif error is not None:
                self._error(error)

        def _error(self, error):
            self.statusBar().showMessage("エラー: " + str(error)[:300], 14000)
            self.activity.setText("エラー: " + str(error)[:160])

        def _construct(self):
            root = QWidget()
            root.setObjectName("outer")
            self.setCentralWidget(root)
            lay = col_layout(root)
            self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
            self.main_splitter.setChildrenCollapsible(False)
            lay.addWidget(self.main_splitter)
            self._sidebar()
            self._center()
            self._inspector()
            self.main_splitter.setSizes([238, 935, 330])
            self.statusBar().setStyleSheet("QStatusBar {background: #14212b; color: #a7c5bb; padding: 4px 14px;}")
            self.statusBar().showMessage("準備完了  ·  ローカル専用  ·  ブラウザは起動しません")

        def _sidebar(self):
            side = frame("sidebar")
            side.setMinimumWidth(188)
            side.setMaximumWidth(320)
            self.main_splitter.addWidget(side)
            lay = col_layout(side, 18, 15)
            lay.addWidget(label("◈  ironasu", kind="logo"))
            lay.addWidget(label("STUDIO   /   DESKTOP  1.7", kind="caption"))
            lay.addSpacing(10)
            lay.addWidget(button("＋ 新しいチャット", self.new_session, primary=True))
            lay.addSpacing(12)
            lay.addWidget(label("会話履歴", kind="section"))
            self.session_list = QListWidget()
            self.session_list.currentRowChanged.connect(self._select_session)
            lay.addWidget(self.session_list, 1)
            lay.addWidget(button("JSONにエクスポート", self.export_chat))
            lay.addWidget(button("会話を削除", self.delete_chat))
            self.history_note = label("履歴は起動中のメモリ内のみ", kind="caption", wrap=True)
            lay.addWidget(self.history_note)

        def _center(self):
            center = QWidget()
            center.setObjectName("workArea")
            center.setMinimumWidth(550)
            self.main_splitter.addWidget(center)
            lay = col_layout(center, 17, 14)
            self.tabs = QTabWidget()
            lay.addWidget(self.tabs)
            self._chat_tab()
            self._library_tab()
            self._agents_tab()
            self._mission_tab()

        def _inspector(self):
            shell = frame("inspector")
            shell.setMinimumWidth(260)
            shell.setMaximumWidth(540)
            self.main_splitter.addWidget(shell)
            outer = col_layout(shell)
            body = QWidget()
            body.setObjectName("inspector")
            lay = col_layout(body, 18, 13)
            lay.addWidget(label("Runtime", kind="title"))
            lay.addWidget(label("接続先", kind="caption"))
            self.provider_combo = QComboBox()
            self.provider_combo.addItems(["auto", *PRESETS.keys()])
            self.provider_combo.setCurrentText(self.provider)
            self.provider_combo.currentTextChanged.connect(self._switch_backend)
            lay.addWidget(self.provider_combo)
            lay.addWidget(button("接続とモデル一覧を更新", self.refresh_models))
            lay.addSpacing(8)
            lay.addWidget(label("使用モデル / ルート", kind="caption"))
            self.model_combo = QComboBox()
            self.model_combo.addItems(["auto", "fast", "quality"])
            self.model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.model_combo.setMinimumContentsLength(13)
            lay.addWidget(self.model_combo)
            self.autoload = QCheckBox("LM Studioモデルを自動ロード")
            self.autoload.setToolTip("ローカルLM Studioのインストール済みモデルのみ／明示的に選択したとき")
            lay.addWidget(self.autoload)
            lay.addSpacing(12)
            lay.addWidget(label("推論設定", kind="section"))
            lay.addWidget(label("Temperature", kind="caption"))
            self.temperature = QDoubleSpinBox()
            self.temperature.setRange(0.0, 1.5)
            self.temperature.setSingleStep(0.05)
            self.temperature.setDecimals(2)
            self.temperature.setValue(0.7)
            lay.addWidget(self.temperature)
            lay.addWidget(label("Top P", kind="caption"))
            self.top_p = QDoubleSpinBox()
            self.top_p.setRange(0.05, 1.0)
            self.top_p.setSingleStep(0.05)
            self.top_p.setValue(0.95)
            lay.addWidget(self.top_p)
            lay.addWidget(label("最大生成トークン", kind="caption"))
            self.max_tokens = QSpinBox()
            self.max_tokens.setRange(1, 32768)
            self.max_tokens.setValue(1024)
            lay.addWidget(self.max_tokens)
            lay.addWidget(label("System prompt", kind="caption"))
            self.system_prompt = QPlainTextEdit()
            self.system_prompt.setPlaceholderText("任意のシステム指示")
            self.system_prompt.setFixedHeight(88)
            lay.addWidget(self.system_prompt)
            self.stream_enabled = QCheckBox("ストリーミング表示")
            self.stream_enabled.setChecked(True)
            lay.addWidget(self.stream_enabled)
            self.show_reasoning = QCheckBox("Reasoningを別表示")
            self.show_reasoning.setChecked(True)
            self.show_reasoning.toggled.connect(self._render_chat)
            lay.addWidget(self.show_reasoning)
            lay.addSpacing(12)
            lay.addWidget(label("直近の推論計測", kind="section"))
            self.perf = QGridLayout()
            self.perf.setHorizontalSpacing(13)
            self.perf.setVerticalSpacing(11)
            self.perf_values = {}
            for idx, (key, title) in enumerate((
                ("ttft_s", "初回受信 TTFT"), ("tokens_per_s", "生成 tok/s"),
                ("reasoning_s", "Reasoning受信"), ("chars_per_s", "文字/秒")
            )):
                cell = frame()
                cl = col_layout(cell, 12, 5)
                cl.addWidget(label(title, kind="caption", wrap=True))
                val = label("—", kind="metric")
                cl.addWidget(val)
                self.perf_values[key] = val
                self.perf.addWidget(cell, idx // 2, idx % 2)
            lay.addLayout(self.perf)
            lay.addWidget(label(
                "TTFTは通信と待機を含む端末観測値。\n"
                "tok/sはモデルがtoken usageを返す場合のみ表示。\n"
                "内部思考の時間やGPU速度を推定しません。",
                kind="caption", wrap=True))
            lay.addSpacing(10)
            self.runtime_hint = label("管理GGUFランタイム: 未確認", kind="caption", wrap=True)
            lay.addWidget(self.runtime_hint)
            lay.addStretch(1)
            outer.addWidget(scroller(body))

        def _chat_tab(self):
            page = QWidget()
            lay = col_layout(page, 0, 12)
            top = row_layout()
            self.chat_title = label("新しいチャット", kind="title", wrap=True)
            top.addWidget(self.chat_title, 1)
            self.connection_label = label("接続を確認中", kind="caption", wrap=True)
            top.addWidget(self.connection_label)
            lay.addLayout(top)
            self.chat_scroll = QScrollArea()
            self.chat_scroll.setWidgetResizable(True)
            self.chat_shell = QWidget()
            self.chat_list = col_layout(self.chat_shell, 10, 11)
            self.chat_list.addStretch(1)
            self.chat_scroll.setWidget(self.chat_shell)
            lay.addWidget(self.chat_scroll, 1)
            composer, cl = box("メッセージ")
            self.prompt = QPlainTextEdit()
            self.prompt.setPlaceholderText("メッセージを入力…  （Ctrl+Enterで送信）")
            self.prompt.setFixedHeight(104)
            cl.addWidget(self.prompt)
            bottom = row_layout()
            self.activity = label("Ctrl+Enterで送信 · データは指定したモデルに送信されます", kind="caption", wrap=True)
            bottom.addWidget(self.activity, 1)
            self.stop_btn = button("停止", self.stop_generation)
            self.stop_btn.setEnabled(False)
            bottom.addWidget(self.stop_btn)
            self.send_btn = button("送信 ↗", self.send, primary=True)
            bottom.addWidget(self.send_btn)
            cl.addLayout(bottom)
            lay.addWidget(composer)
            self.tabs.addTab(page, "チャット")
            self.send_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self.prompt)
            self.send_shortcut.activated.connect(self.send)

        def _library_tab(self):
            page = QWidget()
            base = col_layout(page, 0, 10)
            base.addWidget(label("モデル管理", kind="title"))
            splitter = QSplitter(Qt.Orientation.Vertical)
            splitter.setChildrenCollapsible(False)
            base.addWidget(splitter, 1)
            upper = QSplitter(Qt.Orientation.Horizontal)
            upper.setChildrenCollapsible(False)
            splitter.addWidget(upper)
            hf, left = box("Hugging Face · GGUFをironasuへ直接保存")
            hf.setMinimumWidth(280)
            upper.addWidget(hf)
            search = row_layout()
            self.hf_query = QLineEdit()
            self.hf_query.setPlaceholderText("モデルを検索")
            search.addWidget(self.hf_query, 1)
            search.addWidget(button("検索", self.search_hf))
            left.addLayout(search)
            self.hf_results = QListWidget()
            self.hf_results.setMaximumHeight(95)
            self.hf_results.itemSelectionChanged.connect(self._hf_chosen)
            left.addWidget(self.hf_results)
            repo = row_layout()
            self.hf_repo = QLineEdit()
            self.hf_repo.setPlaceholderText("例: ggml-org/gemma-3-1b-it-GGUF")
            repo.addWidget(self.hf_repo, 1)
            repo.addWidget(button("ファイル一覧", self.list_hf_files))
            left.addLayout(repo)
            self.hf_file_list = QListWidget()
            self.hf_file_list.setMinimumHeight(120)
            left.addWidget(self.hf_file_list, 1)
            left.addWidget(button("選択したGGUFをダウンロード", self.download_native, primary=True))
            left.addWidget(label("新規ダウンロードにはネットワーク接続が必要です。",
                                 kind="caption", wrap=True))
            local, right = box("保存済みGGUF · モデル起動")
            local.setMinimumWidth(290)
            upper.addWidget(local)
            right_buttons = row_layout()
            right_buttons.addWidget(button("一覧更新", self.refresh_native))
            right_buttons.addStretch(1)
            right.addLayout(right_buttons)
            self.native_tree = tree(["モデル", "サイズ"])
            self.native_tree.setMinimumHeight(140)
            right.addWidget(self.native_tree, 1)
            settings = row_layout()
            self.ctx_length = QSpinBox()
            self.ctx_length.setRange(512, 131072)
            self.ctx_length.setSingleStep(512)
            self.ctx_length.setValue(4096)
            settings.addWidget(label("Context"))
            settings.addWidget(self.ctx_length, 1)
            self.gpu_layers = QSpinBox()
            self.gpu_layers.setRange(-1, 999)
            self.gpu_layers.setSpecialValueText("自動")
            self.gpu_layers.setValue(-1)
            settings.addWidget(label("GPU層"))
            settings.addWidget(self.gpu_layers, 1)
            right.addLayout(settings)
            launch = row_layout()
            launch.addWidget(button("GGUFを起動", self.launch_native, primary=True))
            launch.addWidget(button("停止", self.stop_native))
            launch.addWidget(button("削除", self.remove_native))
            right.addLayout(launch)
            self.runtime_status = label("ローカル推論エンジン: 未確認", kind="caption", wrap=True)
            right.addWidget(self.runtime_status)
            upper.setSizes([510, 510])
            lower = QSplitter(Qt.Orientation.Horizontal)
            lower.setChildrenCollapsible(False)
            splitter.addWidget(lower)
            delegated, dl = box("LM Studio / Ollama · ロード済みとインストール済み")
            delegated.setMinimumWidth(280)
            lower.addWidget(delegated)
            provider_row = row_layout()
            self.delegate_provider = QComboBox()
            self.delegate_provider.addItems(["lmstudio", "ollama"])
            self.delegate_provider.currentTextChanged.connect(self.refresh_delegated)
            provider_row.addWidget(self.delegate_provider)
            self.delegate_id = QLineEdit()
            self.delegate_id.setPlaceholderText("モデルIDを入力")
            provider_row.addWidget(self.delegate_id, 1)
            provider_row.addWidget(button("取得", self.download_delegate))
            dl.addLayout(provider_row)
            self.delegate_tree = tree(["モデル", "容量", "状態"])
            dl.addWidget(self.delegate_tree, 1)
            action = row_layout()
            action.addWidget(button("一覧更新", self.refresh_delegated))
            action.addWidget(button("LM Studioでロード", self.load_delegated))
            dl.addLayout(action)
            jobs, jl = box("ダウンロード進捗")
            jobs.setMinimumWidth(270)
            lower.addWidget(jobs)
            self.jobs_tree = tree(["モデル", "状態", "進捗"])
            jl.addWidget(self.jobs_tree, 1)
            jl.addWidget(button("選択ジョブを中断", self.cancel_native_download))
            lower.setSizes([510, 510])
            splitter.setSizes([510, 330])
            self.tabs.addTab(page, "モデル管理")

        def _agents_tab(self):
            page = QWidget()
            base = col_layout(page, 0, 11)
            title_row = row_layout()
            title_row.addWidget(label("コーパス並列エージェント", kind="title"))
            title_row.addStretch()
            title_row.addWidget(label("最大16セッション · 推論同時枠は別管理", kind="caption"))
            base.addLayout(title_row)
            splitter = QSplitter(Qt.Orientation.Vertical)
            splitter.setChildrenCollapsible(False)
            base.addWidget(splitter, 1)
            setup, sl = box("分析指示・コーパス")
            splitter.addWidget(setup)
            sl.addWidget(label("分析する目的", kind="caption"))
            self.agent_objective = QPlainTextEdit()
            self.agent_objective.setPlainText("コーパスの要点・相違点・根拠を整理してください。")
            self.agent_objective.setFixedHeight(65)
            sl.addWidget(self.agent_objective)
            corpus_title = row_layout()
            corpus_title.addWidget(label("コーパス本文（UTF-8 · 最大100万文字）", kind="caption"))
            corpus_title.addStretch()
            corpus_title.addWidget(button("ファイルを読み込む", self.open_corpus))
            sl.addLayout(corpus_title)
            self.agent_text = QPlainTextEdit()
            self.agent_text.setPlaceholderText("ここにテキストを貼り付けるか、ファイルから読み込みます。")
            self.agent_text.setMinimumHeight(95)
            self.agent_text.textChanged.connect(self._agent_edited)
            sl.addWidget(self.agent_text, 1)
            self.agent_file_note = label("ファイルを読み込んだ場合は先頭部分だけをプレビューします。",
                                         kind="caption", wrap=True)
            sl.addWidget(self.agent_file_note)
            controls = row_layout()
            controls.addWidget(label("分割数"))
            self.agent_parts = QComboBox()
            self.agent_parts.addItems(["4", "16"])
            controls.addWidget(self.agent_parts)
            controls.addWidget(label("エージェント"))
            self.agent_workers = QComboBox()
            self.agent_workers.addItems(["1", "2", "4", "8", "12", "16"])
            self.agent_workers.setCurrentText("4")
            controls.addWidget(self.agent_workers)
            controls.addWidget(label("モデル同時"))
            self.agent_slots = QComboBox()
            self.agent_slots.addItems(["1", "2", "4", "8", "16"])
            self.agent_slots.setCurrentText("2")
            controls.addWidget(self.agent_slots)
            controls.addStretch()
            sl.addLayout(controls)
            actions = row_layout()
            self.agent_start_btn = button("並列分析を開始", self.start_agents, primary=True)
            actions.addWidget(self.agent_start_btn)
            self.agent_stop_btn = button("停止", self.stop_agents)
            self.agent_stop_btn.setEnabled(False)
            actions.addWidget(self.agent_stop_btn)
            self.agent_progress_bar = QProgressBar()
            self.agent_progress_bar.setRange(0, 100)
            self.agent_progress_bar.setValue(0)
            actions.addWidget(self.agent_progress_bar, 1)
            sl.addLayout(actions)
            self.agent_status = label("未実行", kind="caption", wrap=True)
            sl.addWidget(self.agent_status)
            report, rl = box("分析結果・エビデンス")
            splitter.addWidget(report)
            self.agent_output = QPlainTextEdit()
            self.agent_output.setReadOnly(True)
            self.agent_output.setPlaceholderText("結果はここに表示されます。")
            rl.addWidget(self.agent_output, 1)
            rr = row_layout()
            rr.addWidget(label("重複再利用・モデル呼び出し・SHA-256索引をレポートに収録", kind="caption", wrap=True), 1)
            rr.addWidget(button("レポートをJSON保存", self.export_agent_report))
            rl.addLayout(rr)
            splitter.setSizes([400, 420])
            self.tabs.addTab(page, "エージェント ×16")

        def _mission_tab(self):
            page = QWidget()
            lay = col_layout(page, 0, 12)
            title = row_layout()
            title.addWidget(label("Mission Controller · 開発任務", kind="title"))
            title.addStretch()
            title.addWidget(label("最大16エージェント / 明示的な操作権限 / SQLite復旧",
                                  kind="caption", wrap=True))
            lay.addLayout(title)
            vertical = QSplitter(Qt.Orientation.Vertical)
            vertical.setChildrenCollapsible(False)
            lay.addWidget(vertical, 1)
            form, fl = box("01 · 任務と作業領域")
            vertical.addWidget(form)
            fl.addWidget(label("達成したい目標（計画 → 実装 → 検証 → 修復）", kind="caption"))
            self.mission_goal = QPlainTextEdit()
            self.mission_goal.setPlaceholderText("例：リポジトリの既存テストを確認し、原因を修正してください。")
            self.mission_goal.setFixedHeight(95)
            fl.addWidget(self.mission_goal)
            row = row_layout()
            row.addWidget(label("作業フォルダー", kind="caption"))
            self.mission_workspace = QLineEdit(str(Path.home()))
            row.addWidget(self.mission_workspace, 1)
            row.addWidget(button("参照…", self._mission_choose_folder))
            fl.addLayout(row)
            row = row_layout()
            row.addWidget(label("エージェント", kind="caption"))
            self.mission_workers = QSpinBox()
            self.mission_workers.setRange(1, 16)
            self.mission_workers.setValue(4)
            row.addWidget(self.mission_workers)
            row.addWidget(label("モデル同時", kind="caption"))
            self.mission_slots = QSpinBox()
            self.mission_slots.setRange(1, 16)
            self.mission_slots.setValue(2)
            row.addWidget(self.mission_slots)
            row.addWidget(label("修復上限", kind="caption"))
            self.mission_repairs = QSpinBox()
            self.mission_repairs.setRange(0, 5)
            self.mission_repairs.setValue(2)
            row.addWidget(self.mission_repairs)
            row.addStretch()
            fl.addLayout(row)
            row = row_layout()
            self.mission_writes = QCheckBox("ファイル変更を許可")
            self.mission_tests = QCheckBox("検証コマンドの実行を許可")
            row.addWidget(self.mission_writes)
            row.addWidget(self.mission_tests)
            row.addStretch()
            fl.addLayout(row)
            fl.addWidget(label(
                "既定は読み取り専用。編集許可時は作業領域の競合を防ぐため"
                "1ワーカーずつ実行します。テストはプロジェクト内のコードを実行します。",
                kind="caption", wrap=True))
            row = row_layout()
            row.addWidget(label("確認したい成果物（相対パス／カンマ区切り）", kind="caption"))
            self.mission_files = QLineEdit()
            self.mission_files.setPlaceholderText("例：src/fixed.py, tests/test_fixed.py")
            row.addWidget(self.mission_files, 1)
            fl.addLayout(row)
            actions = row_layout()
            self.mission_start = button("新しい任務を開始", self._mission_new, primary=True)
            actions.addWidget(self.mission_start)
            self.mission_resume = button("保存した任務を再開", self._mission_resume)
            actions.addWidget(self.mission_resume)
            self.mission_stop = button("停止", self._mission_stop)
            self.mission_stop.setEnabled(False)
            actions.addWidget(self.mission_stop)
            actions.addStretch()
            fl.addLayout(actions)

            results, rl = box("02 · 実行ログ・検証結果・復旧")
            vertical.addWidget(results)
            row = row_layout()
            row.addWidget(label("任務ID", kind="caption"))
            self.mission_id = QLineEdit()
            self.mission_id.setPlaceholderText("再開する任務のID")
            row.addWidget(self.mission_id, 1)
            row.addWidget(button("保存済みを表示", self._mission_refresh))
            rl.addLayout(row)
            self.mission_recent = QListWidget()
            self.mission_recent.setMaximumHeight(90)
            self.mission_recent.currentItemChanged.connect(self._mission_pick)
            rl.addWidget(self.mission_recent)
            self.mission_status = label(
                "未実行 · 結果はSQLiteに保存。停止後の再開には同じ作業フォルダーと任務IDを指定。",
                kind="caption", wrap=True)
            rl.addWidget(self.mission_status)
            self.mission_output = QPlainTextEdit()
            self.mission_output.setReadOnly(True)
            self.mission_output.setPlaceholderText(
                "各タスクの実行状況、検証の合否、修復回数をここに表示します。"
                "LLMが「完了」と返すだけでは成功扱いにしません。")
            rl.addWidget(self.mission_output, 1)
            row = row_layout()
            row.addWidget(label(
                "Git/テスト/生成物で確認。外部サービス操作や任意シェル実行は許可しません。",
                kind="caption", wrap=True), 1)
            row.addWidget(button("レポートJSON保存", self._mission_export))
            rl.addLayout(row)
            vertical.setSizes([430, 370])
            self.tabs.addTab(page, "ミッション")
            # Do not create .ironasu files just by opening a UI tab.
            # Explicit refresh / mission creation handles workspace state.

        def _mission_choose_folder(self):
            result = QFileDialog.getExistingDirectory(
                self, "任務の作業フォルダー", self.mission_workspace.text())
            if result:
                self.mission_workspace.setText(result)
                self._mission_refresh()

        def _mission_refresh(self):
            try:
                folder = Path(self.mission_workspace.text()).expanduser().resolve(strict=True)
                if not folder.is_dir():
                    raise ValueError("作業フォルダーが存在しません")
                self.mission_recent.clear()
                if not (folder / ".ironasu" / "missions.sqlite").is_file():
                    return
                rows = MissionJournal(folder).recent()
                for item in rows:
                    text = f"{item['id']} · {item['state']} · {item['goal'][:58]}"
                    self.mission_recent.addItem(text)
                    self.mission_recent.item(self.mission_recent.count()-1).setData(
                        Qt.ItemDataRole.UserRole, item["id"])
            except (OSError, ValueError) as exc:
                self.mission_status.setText("保存済み任務一覧: " + str(exc)[:130])

        def _mission_pick(self, item, _previous):
            if item is not None and not self.mission_busy:
                self.mission_id.setText(item.data(Qt.ItemDataRole.UserRole))

        def _mission_new(self):
            self._mission_execute(False)

        def _mission_resume(self):
            self._mission_execute(True)

        def _mission_execute(self, resume):
            if self.mission_busy:
                return
            goal = self.mission_goal.toPlainText().strip()
            identifier = self.mission_id.text().strip() if resume else uuid4().hex[:16]
            if resume and not identifier:
                self._error("再開する任務IDを選択・入力してください。")
                return
            if not resume and not goal:
                self._error("任務の目標を入力してください。")
                return
            folder = Path(self.mission_workspace.text()).expanduser()
            if not folder.is_dir():
                self._error("作業フォルダーが存在しません。")
                return
            permits_writes = self.mission_writes.isChecked()
            permits_tests = self.mission_tests.isChecked()
            if permits_writes or permits_tests:
                accepted = QMessageBox.question(
                    self, "権限の確認",
                    "この任務に許可する操作:\n"
                    + ("・作業フォルダー内のファイル編集\n" if permits_writes else "")
                    + ("・PythonテストやGit検査の実行（プロジェクトのコードが動きます）\n"
                       if permits_tests else "")
                    + "\n明示的に許可しますか？",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No)
                if accepted != QMessageBox.StandardButton.Yes:
                    return
            endpoint = self.mux.endpoint().get("url")
            if not isinstance(endpoint, str) or not endpoint:
                self._error("接続先の推論エンジンを先に選択してください。")
                return
            from urllib.parse import urlsplit
            if urlsplit(endpoint).hostname not in ("127.0.0.1", "localhost", "::1"):
                self._error("GUIのミッション機能はローカル推論エンジンのみ使用できます。")
                return
            selected = self.model_combo.currentText()
            installed = [self.model_combo.itemText(i) for i in range(self.model_combo.count())
                         if self.model_combo.itemText(i) not in ("auto", "fast", "quality")]
            model = selected if selected in installed else installed[0] if installed else None
            if not model:
                self._error("使用するローカルモデルを選択してください。")
                return
            api_key = self.config.backend_key
            if self.managed.model_id() == model and self.managed._backend is not None:
                api_key = self.managed._backend.config.backend_key
            artifact_paths = tuple(x.strip() for x in
                                   self.mission_files.text().split(",") if x.strip())
            self.mission_cancel = Event()
            self.mission_busy = True
            self.mission_id.setText(identifier)
            self.mission_start.setEnabled(False)
            self.mission_resume.setEnabled(False)
            self.mission_stop.setEnabled(True)
            self.mission_status.setText("実行中: " + identifier)
            self.mission_output.appendPlainText(
                "\n▶ " + ("再開" if resume else "新規任務") + ": " + identifier)
            workers, slots, repairs = (
                self.mission_workers.value(), self.mission_slots.value(),
                self.mission_repairs.value())
            stop = self.mission_cancel
            def work():
                mission = MissionController(
                    folder, endpoint=endpoint, model=model, api_key=api_key)
                return mission.run(
                    goal=None if resume else goal, mission_id=identifier,
                    resume=resume, workers=workers, model_slots=slots,
                    max_repairs=repairs, allow_writes=permits_writes,
                    allow_tests=permits_tests, checks=("compileall", "git_diff_check"),
                    accept_files=artifact_paths, cancel=stop,
                    progress=lambda e: self.bridge.mission_progress.emit(e))
            self._run(work, self._mission_done, self._mission_failed)

        def _mission_event(self, event):
            if self._closed:
                return
            row = f"[{event['kind']}] {event['detail']}"
            self.mission_output.appendPlainText(row)
            self.mission_status.setText(row[:160])

        def _mission_done(self, report):
            self.mission_busy = False
            self.mission_report = report
            self.mission_start.setEnabled(True)
            self.mission_resume.setEnabled(True)
            self.mission_stop.setEnabled(False)
            self.mission_status.setText(
                ("検証成功" if report.verified else "未検証／要対応") +
                f" · {report.state} · 修復{report.repairs}回")
            self.mission_output.appendPlainText(
                "\n【検証結果】\n" + report.reason + "\n" +
                "\n".join(f"{t['task_id']}: {t['state']} {t['error'][:140]}"
                          for t in report.tasks) + "\n" +
                "\n".join(f"{c['kind']}: {'PASS' if c['passed'] else 'FAIL'}"
                          for c in report.checks))
            self._mission_refresh()

        def _mission_failed(self, error):
            self.mission_busy = False
            self.mission_start.setEnabled(True)
            self.mission_resume.setEnabled(True)
            self.mission_stop.setEnabled(False)
            self._error(error)

        def _mission_stop(self):
            self.mission_cancel.set()
            self.mission_stop.setEnabled(False)
            self.mission_status.setText("停止要求済み · 実行中のモデル呼び出しは完了待ち")

        def _mission_export(self):
            identifier = self.mission_id.text().strip()
            if not identifier:
                self._error("任務IDを選んでください。")
                return
            path, _ = QFileDialog.getSaveFileName(
                self, "任務レポート保存", "ironasu-mission.json", "JSON (*.json)")
            if not path:
                return
            folder = Path(self.mission_workspace.text())
            endpoint = self.mux.endpoint().get("url") or "http://127.0.0.1:1234/v1"
            model = self.model_combo.currentText() or "unknown"
            def dump():
                journal = MissionJournal(folder)
                report = MissionController(
                    folder, endpoint=endpoint, model=model).report(identifier)
                Path(path).write_text(json.dumps({
                    "report": report.as_dict(), "events": journal.events(identifier, 500),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                return path
            self._run(dump, lambda name: self.mission_status.setText(
                "レポートを保存: " + name), self._error)

        def _render_sessions(self):
            self.session_list.blockSignals(True)
            self.session_list.clear()
            for s in self.sessions:
                self.session_list.addItem(s.title[:70])
            self.session_list.setCurrentRow(self.session_index)
            self.session_list.blockSignals(False)

        def _select_session(self, index):
            if self.chat_busy or index < 0 or index >= len(self.sessions):
                return
            self.session_index = index
            self._render_chat()

        def new_session(self):
            if self.chat_busy:
                return
            self.sessions.insert(0, Session())
            self.session_index = 0
            self._render_sessions()
            self._render_chat()
            self.tabs.setCurrentIndex(0)

        def delete_chat(self):
            if self.chat_busy:
                return
            if QMessageBox.question(self, "会話の削除", "選択中の会話を削除しますか？") != QMessageBox.StandardButton.Yes:
                return
            self.sessions.pop(self.session_index)
            if not self.sessions:
                self.sessions.append(Session())
            self.session_index = 0
            self._render_sessions()
            self._render_chat()

        def export_chat(self):
            selected, _ = QFileDialog.getSaveFileName(self, "会話のエクスポート", "ironasu-chat.json", "JSON (*.json)")
            if not selected:
                return
            session = self.sessions[self.session_index]
            doc = {"title": session.title, "messages": [
                {k: v for k, v in m.items() if k in ("role", "content", "reasoning", "status")}
                for m in session.messages]}
            try:
                Path(selected).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError as exc:
                self._error(exc)

        def _render_chat(self):
            scroll = self.chat_scroll.verticalScrollBar()
            at_bottom = scroll.value() >= scroll.maximum() - 30
            self.chat_title.setText(self.sessions[self.session_index].title)
            while self.chat_list.count():
                item = self.chat_list.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self._message_widgets = []
            for msg in self.sessions[self.session_index].messages:
                widget = MessageCard(msg)
                widget.update_message(msg, self.show_reasoning.isChecked())
                self.chat_list.addWidget(widget)
                self._message_widgets.append(widget)
            self.chat_list.addStretch(1)
            if at_bottom:
                QTimer.singleShot(0, lambda: scroll.setValue(scroll.maximum()))

        def _stream_delta(self, sid, index, typ, text):
            if not (0 <= sid < len(self.sessions)) or index >= len(self.sessions[sid].messages):
                return
            msg = self.sessions[sid].messages[index]
            key = "reasoning" if typ == "reasoning" else "content"
            msg[key] = msg.get(key, "") + text
            if sid == self.session_index and index < len(self._message_widgets):
                self._message_widgets[index].update_message(msg, self.show_reasoning.isChecked())
                scroll = self.chat_scroll.verticalScrollBar()
                if scroll.value() >= scroll.maximum() - 60:
                    QTimer.singleShot(0, lambda: scroll.setValue(scroll.maximum()))

        def _generation_done(self, sid, index, status, stats, failure):
            if 0 <= sid < len(self.sessions) and index < len(self.sessions[sid].messages):
                msg = self.sessions[sid].messages[index]
                msg["status"] = {"completed": "完了", "stopped": "停止", "failed": "エラー"}.get(status, status)
                if status != "completed":
                    msg["failed"] = True
                if sid == self.session_index and index < len(self._message_widgets):
                    self._message_widgets[index].update_message(msg, self.show_reasoning.isChecked())
            self.chat_busy = False
            self.send_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.provider_combo.setEnabled(True)
            if stats:
                for key, node in self.perf_values.items():
                    value = stats.get(key)
                    unit = "s" if key in ("ttft_s", "reasoning_s") else ""
                    node.setText("—" if value is None else f"{value:.2f}{unit}")
            if failure:
                self._error(failure)
            else:
                self.activity.setText("完了" if status == "completed" else "生成を停止しました")

        def _messages_payload(self, sid):
            messages = []
            system = self.system_prompt.toPlainText().strip()
            if system:
                messages.append({"role": "system", "content": system})
            for msg in self.sessions[sid].messages:
                if msg.get("failed") or not msg.get("content"):
                    continue
                if msg.get("role") in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})
            mode, direct = normalized_model_selection(self.model_combo.currentText())
            payload = {"model": "ironasu", "messages": messages, "stream": self.stream_enabled.isChecked(),
                       "temperature": self.temperature.value(), "top_p": self.top_p.value(),
                       "max_tokens": self.max_tokens.value(), "x_ironasu_mode": mode,
                       "x_ironasu_cache": True}
            if direct:
                payload["x_ironasu_model"] = direct
            return payload

        def send(self):
            if self.chat_busy:
                return
            prompt = self.prompt.toPlainText().strip()
            if not prompt:
                return
            sid = self.session_index
            session = self.sessions[sid]
            if session.title == "新しいチャット":
                session.title = " ".join(prompt.split())[:55]
            session.messages.append({"role": "user", "content": prompt})
            session.messages.append({"role": "assistant", "content": "", "reasoning": ""})
            index = len(session.messages) - 1
            self.prompt.clear()
            payload = self._messages_payload(sid)
            self.chat_cancel = Event()
            stop = self.chat_cancel
            self.chat_busy = True
            self.send_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.provider_combo.setEnabled(False)
            self._render_sessions()
            self._render_chat()
            self.activity.setText("モデルから応答を受信しています…")
            explicit = payload.get("x_ironasu_model")
            auto_load = (self.autoload.isChecked() and explicit and not explicit.startswith("ironasu-local-"))
            engine = self.engine
            models = self.models
            def worker():
                stats = GenerationStats()
                outcome, failure = "completed", None
                try:
                    if auto_load:
                        endpoint = engine.backend.endpoint()
                        if endpoint.get("url") != "http://127.0.0.1:1234/v1":
                            raise InferenceError("LM Studio自動ロードはローカル1234番のみ", 409)
                        models.ensure_loaded("lmstudio", explicit)
                    if stop.is_set():
                        raise InterruptedError("Stopped")
                    if payload["stream"]:
                        with engine.stream(payload) as response:
                            with self.stream_lock:
                                self.active_stream = response
                            decoder = SSEEvents()
                            for line in response:
                                if stop.is_set():
                                    raise InterruptedError("Stopped")
                                for chunk in decoder.feed(line):
                                    for typ, text in stats.ingest(chunk, time.perf_counter()):
                                        self.bridge.stream_delta.emit(sid, index, typ, text)
                            for typ, text in stats.append_content("", time.perf_counter(), flush=True):
                                self.bridge.stream_delta.emit(sid, index, typ, text)
                    else:
                        data = engine.complete(payload)
                        first = (data.get("choices") or [{}])[0].get("message") or {}
                        reason = extract_reasoning(first)
                        if reason:
                            self.bridge.stream_delta.emit(sid, index, "reasoning", reason)
                        for typ, text in stats.append_content(extract_text(first), time.perf_counter(), flush=True):
                            self.bridge.stream_delta.emit(sid, index, typ, text)
                        usage = data.get("usage") or {}
                        if type(usage.get("completion_tokens")) is int:
                            stats.tokens = usage["completion_tokens"]
                        stats.first = None
                        stats.first_reasoning = None
                    if stop.is_set():
                        raise InterruptedError("Stopped")
                except Exception as exc:
                    outcome = "stopped" if stop.is_set() else "failed"
                    if outcome == "failed":
                        failure = exc
                finally:
                    with self.stream_lock:
                        self.active_stream = None
                    self.bridge.generation_done.emit(sid, index, outcome,
                                                    stats.finish(time.perf_counter()), failure)
            Thread(target=worker, daemon=True, name="ironasu-qt-chat").start()

        def stop_generation(self):
            self.chat_cancel.set()
            with self.stream_lock:
                stream = self.active_stream
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

        def _switch_backend(self, value):
            if self.chat_busy or not value:
                return
            self.provider = value
            self.backend = AutoHTTPBackend(self.config) if value == "auto" else HTTPBackend(
                replace(self.config, backend_url=PRESETS[value]))
            self.mux = RuntimeBackendMux(self.backend, self.managed)
            self.engine = AdaptiveEngine(self.config, self.mux)
            self.refresh_models()

        def refresh_models(self):
            if self._closed:
                return
            mux = self.mux
            def done(data):
                names, endpoint = data
                old = self.model_combo.currentText()
                preferred = self._last_runtime_model if old == "auto" and self._last_runtime_model else None
                with QSignalBlocker(self.model_combo):
                    self.model_combo.clear()
                    self.model_combo.addItems(["auto", "fast", "quality", *(names or [])])
                    values = ["auto", "fast", "quality", *(names or [])]
                    self.model_combo.setCurrentText(preferred if preferred in values else old if old in values else "auto")
                source = endpoint.get("provider") or "未接続"
                self.connection_label.setText(f"{source} · {len(names or [])} モデル" if names is not None
                                              else f"{source} · サーバー未検出")
            self._run(lambda: (mux.list_models(), mux.endpoint()), done)

        @staticmethod
        def _bytes(size):
            if type(size) is not int or size < 0:
                return "—"
            v = float(size)
            for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
                if v < 1024 or unit == "TiB":
                    return f"{v:.1f} {unit}"
                v /= 1024
            return "—"

        def _poll(self):
            try:
                state = self.managed.status()
                signature = (state.get("state"), state.get("model_id"), state.get("detail"))
                if signature != self._last_runtime:
                    self._last_runtime = signature
                    self.runtime_status.setText("管理モデル: " + state.get("state", "-") + " · " + state.get("detail", ""))
                    self.runtime_hint.setText("管理GGUF: " + state.get("state", "-"))
                if state.get("model_id") != self._last_runtime_model:
                    previous = self._last_runtime_model
                    self._last_runtime_model = state.get("model_id")
                    self.refresh_models()
                    if previous is None and state.get("model_id"):
                        self.statusBar().showMessage("GGUFの起動が完了。モデル一覧から選択できます。", 9000)
                self._render_jobs()
            except Exception as exc:
                self._error(exc)

        def _render_jobs(self):
            jobs = self.native.jobs() + self.models.list_jobs()
            sig = repr([(job["id"], job["state"], job.get("completed_bytes")) for job in jobs])
            if sig == self._job_signature:
                return
            self._job_signature = sig
            self.jobs_tree.clear()
            for job in jobs[-30:]:
                total = job.get("total_bytes")
                done = job.get("completed_bytes")
                p = f"{100 * done / total:.1f}%" if type(total) is int and total > 0 and type(done) is int else "—"
                item = QTreeWidgetItem([str(job.get("model", ""))[:90], job.get("state", ""), p])
                item.setData(0, Qt.ItemDataRole.UserRole, job["id"])
                self.jobs_tree.addTopLevelItem(item)
            if any(j.get("state") == "completed" for j in jobs):
                self.refresh_native()

        def search_hf(self):
            value = self.hf_query.text().strip()
            def done(rows):
                self.native_results = rows
                self.hf_results.clear()
                for row in rows:
                    self.hf_results.addItem(row["id"])
            self._run(lambda: self.native.search(value), done)

        def _hf_chosen(self):
            index = self.hf_results.currentRow()
            if 0 <= index < len(self.native_results):
                self.hf_repo.setText(self.native_results[index]["id"])

        def list_hf_files(self):
            repo = self.hf_repo.text().strip()
            self.native_files = []
            self.hf_file_list.clear()
            def done(rows):
                self.native_files = rows
                self.hf_file_list.clear()
                for row in rows:
                    self.hf_file_list.addItem(row["path"] + "  ·  " + self._bytes(row.get("size_bytes")))
            self._run(lambda: self.native.files(repo), done)

        def download_native(self):
            index = self.hf_file_list.currentRow()
            if not 0 <= index < len(self.native_files):
                self._error("ダウンロードするGGUFを選択してください。")
                return
            item = self.native_files[index]
            if QMessageBox.question(self, "モデルのダウンロード",
                                    f"{item['path']}\n{self._bytes(item.get('size_bytes'))}\n取得しますか？") != QMessageBox.StandardButton.Yes:
                return
            repo = self.hf_repo.text().strip()
            self._run(lambda: self.native.start(repo, item["path"]),
                      lambda _job: self._render_jobs())

        def refresh_native(self):
            if self._closed:
                return
            def done(rows):
                selected = self.native_tree.currentItem()
                old = selected.data(0, Qt.ItemDataRole.UserRole) if selected else None
                self.native_tree.clear()
                self.native_installed = rows
                for item in rows:
                    cell = QTreeWidgetItem([item["label"], self._bytes(item.get("size_bytes"))])
                    cell.setData(0, Qt.ItemDataRole.UserRole, item["id"])
                    self.native_tree.addTopLevelItem(cell)
                    if cell.data(0, Qt.ItemDataRole.UserRole) == old:
                        self.native_tree.setCurrentItem(cell)
            self._run(self.native.installed, done)

        def _selected_native(self):
            item = self.native_tree.currentItem()
            if not item:
                self._error("保存済みGGUFを選択してください。")
                return None
            return next((row for row in self.native_installed
                         if row["id"] == item.data(0, Qt.ItemDataRole.UserRole)), None)

        def launch_native(self):
            item = self._selected_native()
            if item is None:
                return
            context = self.ctx_length.value()
            gpu = self.gpu_layers.value()
            self._run(lambda: self.managed.start(item["id"], context_length=context,
                                                  gpu_layers=None if gpu == -1 else gpu),
                      lambda _status: self.statusBar().showMessage("GGUFの整合性確認と起動を開始", 7000))

        def stop_native(self):
            self._run(self.managed.stop, lambda _status: self.refresh_models())

        def remove_native(self):
            item = self._selected_native()
            if item is None:
                return
            state = self.managed.status()
            if state.get("installed_id") == item["id"] and state.get("state") in ("starting", "running"):
                self._error("GGUFを停止してから削除してください。")
                return
            if QMessageBox.question(self, "削除", f"{item['label']}\nローカルGGUFファイルを削除しますか？") != QMessageBox.StandardButton.Yes:
                return
            self._run(lambda: self.native.remove(item["id"]), lambda _res: self.refresh_native())

        def cancel_native_download(self):
            selected = self.jobs_tree.currentItem()
            if not selected:
                return
            key = selected.data(0, Qt.ItemDataRole.UserRole)
            if not str(key).startswith("native_"):
                self._error("この画面から中断できるのはironasuによるGGUF取得のみです。")
                return
            self._run(lambda: self.native.cancel(key), lambda _res: self._render_jobs())

        def refresh_delegated(self):
            provider = self.delegate_provider.currentText()
            def done(rows):
                if self.delegate_provider.currentText() != provider:
                    return
                self.delegated_installed = rows
                self.delegate_tree.clear()
                for row in rows:
                    cell = QTreeWidgetItem([row["id"], self._bytes(row.get("size_bytes")),
                                            "ロード済み" if row.get("loaded") else "—"])
                    cell.setData(0, Qt.ItemDataRole.UserRole, row["id"])
                    self.delegate_tree.addTopLevelItem(cell)
            self._run(lambda: self.models.installed(provider), done)

        def download_delegate(self):
            provider = self.delegate_provider.currentText()
            model = self.delegate_id.text().strip()
            if not model:
                self._error("モデルIDを入力してください。")
                return
            if QMessageBox.question(self, "モデルを取得", f"{provider}に{model}の取得を依頼しますか？") != QMessageBox.StandardButton.Yes:
                return
            self._run(lambda: self.models.start(provider, model), lambda _res: self._render_jobs())

        def load_delegated(self):
            if self.delegate_provider.currentText() != "lmstudio":
                self._error("手動ロードはLM Studioに対応しています。")
                return
            item = self.delegate_tree.currentItem()
            if not item:
                self._error("モデルを選択してください。")
                return
            key = item.data(0, Qt.ItemDataRole.UserRole)
            self._run(lambda: self.models.load("lmstudio", key),
                      lambda _res: (self.refresh_models(), self.refresh_delegated()))

        def _agent_edited(self):
            if not self._agent_loading:
                self.agent_file_text = None
                self.agent_file_path = ""
                self.agent_file_note.setText("入力欄のテキストを分析します。")

        def open_corpus(self):
            file, _ = QFileDialog.getOpenFileName(self, "UTF-8コーパスを開く", "",
                    "テキスト (*.txt *.md *.csv *.json *.log);;すべてのファイル (*)")
            if not file:
                return
            try:
                path = Path(file)
                if not path.is_file() or path.stat().st_size > 4 * MAX_CORPUS_CHARS:
                    raise ValueError("ファイルが大きすぎるかファイルではありません。")
                data = path.read_text(encoding="utf-8-sig")
                if len(data) > MAX_CORPUS_CHARS:
                    raise ValueError("最大100万文字です。")
            except (OSError, UnicodeError, ValueError) as exc:
                self._error(exc)
                return
            self.agent_file_text = data
            self.agent_file_path = file
            self._agent_loading = True
            self.agent_text.setPlainText(data[:6000])
            self._agent_loading = False
            self.agent_file_note.setText(f"ファイルから{len(data):,}文字を読み込み（画面には冒頭6000文字）")

        def _agent_corpus(self):
            return self.agent_file_text if self.agent_file_text is not None else self.agent_text.toPlainText()

        def start_agents(self):
            if self.agent_busy:
                return
            corpus = self._agent_corpus()
            objective = self.agent_objective.toPlainText().strip()
            parts = int(self.agent_parts.currentText())
            workers = int(self.agent_workers.currentText())
            slots = min(int(self.agent_slots.currentText()), self.config.max_concurrency)
            mode, name = normalized_model_selection(self.model_combo.currentText())
            model = name or mode
            try:
                if not corpus.strip() or not objective:
                    raise ValueError("コーパスと分析目的を入力してください。")
                if not 1 <= slots <= 16:
                    raise ValueError("モデル同時実行枠が不正です。")
                self.agent_total_segments = sum(len(split_segments(x)) for x in split_corpus(corpus, parts))
            except ValueError as exc:
                self._error(exc)
                return
            self.agent_cancel = Event()
            stop = self.agent_cancel
            self.agent_busy = True
            self.agent_start_btn.setEnabled(False)
            self.agent_stop_btn.setEnabled(True)
            self.agent_progress_bar.setValue(0)
            self.agent_status.setText(f"{parts}分割 · {workers}エージェント · モデル同時{slots}枠")
            self.agent_output.clear()
            self.agent_report = None
            engine = self.engine
            def job():
                return CorpusAgentTeam(engine).run(
                    corpus, objective, partitions=parts, workers=workers, model_slots=slots,
                    model=model, cancel=stop,
                    progress=lambda i, j, total, completed: self.bridge.corpus_progress.emit(
                        i, j, total, completed))
            def done(report):
                self.agent_report = report
                self.agent_busy = False
                self.agent_start_btn.setEnabled(True)
                self.agent_stop_btn.setEnabled(False)
                self.agent_progress_bar.setValue(100 if report.complete else self.agent_progress_bar.value())
                self.agent_status.setText(
                    f"{'完了' if report.complete else '不完全／中断'} · "
                    f"{report.total_segments}チャンク · 固有{report.unique_segments} · "
                    f"再利用{report.cache_hits} · モデル呼び出し{report.model_calls}")
                chunks = ["【統合結果】", report.synthesis or "完全な統合結果はありません", "",
                          "【実行計測】", f"原文文字数: {report.corpus_chars:,}",
                          f"総チャンク数: {report.total_segments}",
                          f"固有チャンク数: {report.unique_segments}",
                          f"重複再利用: {report.cache_hits}",
                          f"モデル呼び出し: {report.model_calls}", ""]
                for part in report.parts:
                    chunks.extend([f"【エージェント {part.index+1:02d}】 {part.status}",
                                   part.error or "", part.summary or "—", ""])
                chunks.append("【原文照合索引：先頭30件／全件はJSON】")
                for ref in report.evidence[:30]:
                    chunks.append(f"{ref.label}  文字{ref.start}-{ref.end}  SHA256:{ref.sha256[:16]}…")
                self.agent_output.setPlainText("\n".join(chunks))
            def fail(exc):
                self.agent_busy = False
                self.agent_start_btn.setEnabled(True)
                self.agent_stop_btn.setEnabled(False)
                self._error(exc)
            self._run(job, done, fail)

        def _agent_progress(self, i, j, total, completed):
            if not self.agent_busy:
                return
            self.agent_status.setText(f"分析中 · 区画{i + 1}  {j}/{total} · 合計{completed}/{self.agent_total_segments}")
            if self.agent_total_segments:
                self.agent_progress_bar.setValue(min(99, int(100 * completed / self.agent_total_segments)))

        def stop_agents(self):
            self.agent_cancel.set()
            self.agent_stop_btn.setEnabled(False)
            self.agent_status.setText("停止要求を送信済み（進行中のモデル呼び出しは完了待ち）")

        def export_agent_report(self):
            if self.agent_report is None:
                self._error("先にエージェント分析を実行してください。")
                return
            output, _ = QFileDialog.getSaveFileName(self, "分析結果を保存", "ironasu-agents.json", "JSON (*.json)")
            if not output:
                return
            report = self.agent_report
            data = {
                "complete": report.complete, "partitions": report.partitions,
                "workers": report.workers, "model_slots": report.model_slots,
                "corpus_chars": report.corpus_chars, "total_segments": report.total_segments,
                "unique_segments": report.unique_segments, "cache_hits": report.cache_hits,
                "model_calls": report.model_calls, "synthesis": report.synthesis,
                "parts": [x.__dict__ for x in report.parts],
                "evidence": [x.__dict__ for x in report.evidence],
            }
            try:
                Path(output).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except OSError as exc:
                self._error(exc)

        def closeEvent(self, event):
            self._closed = True
            self.chat_cancel.set()
            self.agent_cancel.set()
            self.mission_cancel.set()
            self.poll_timer.stop()
            with self.stream_lock:
                stream = self.active_stream
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
            self.managed.stop()
            event.accept()

    return QApplication, MainWindow, QFont, QFontDatabase


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.self_test:
        self_test(args.self_test)
        return 0
    try:
        config, backend = configure(args)
    except ValueError as exc:
        parser().error(str(exc))
    if args.doctor:
        models = backend.list_models()
        endpoint = backend.endpoint() if hasattr(backend, "endpoint") else {
            "provider": args.provider, "url": config.backend_url,
        }
        print(json.dumps({
            "connected": models is not None,
            "provider": endpoint.get("provider"),
            "api_base": endpoint.get("url"),
            "models": models or [],
            "fast_model": config.fast_model or (models[0] if models else None),
            "quality_model": config.quality_model or (models[0] if models else None),
        }, ensure_ascii=False, indent=2))
        return 0 if models else 1
    try:
        QApplication, MainWindow, QFont, QFontDatabase = build_ui()
    except ImportError as exc:
        raise SystemExit("Qt6 desktop cannot load required modules: " + str(exc) + ". Install: pip install 'ironasu[desktop]'") from exc
    app = QApplication.instance() or QApplication([])
    # Qt6 automatically handles Windows per-monitor DPI. No fixed tk/scaling.
    app.setStyle("Fusion")
    families = set(QFontDatabase.families())
    japanese_fonts = ("Yu Gothic UI", "BIZ UDPGothic", "Meiryo UI", "Meiryo", "Noto Sans CJK JP")
    name = next((f for f in japanese_fonts if f in families), "Segoe UI")
    app.setFont(QFont(name, 10))
    app.setStyleSheet(THEME)
    # The frozen Windows executable includes our own deterministic icon.
    if getattr(__import__("sys"), "frozen", False):
        from PySide6.QtGui import QIcon
        icon_path = Path(__import__("sys")._MEIPASS) / "ironasu.ico"
        if icon_path.is_file():
            app.setWindowIcon(QIcon(str(icon_path)))
    if args.smoke_gui:
        class FakeBackend:
            def list_models(self):
                return ["local-test-model"]
            def endpoint(self):
                return {"provider": "fixture", "url": "http://127.0.0.1:1234/v1"}
            def complete(self, payload):
                return {"choices": [{"message": {"content": "ok"}}]}
        window = MainWindow(Config(auto_models=True, fast_model="", quality_model=""),
                            FakeBackend(), smoke=True)
        window.show()
        app.processEvents()
        screenshot = os.environ.get("IRONASU_QT_SCREENSHOT")
        if screenshot:
            root = Path(screenshot)
            for tab, suffix in ((0, ""), (1, "-models"), (2, "-agents"), (3, "-mission")):
                window.tabs.setCurrentIndex(tab)
                app.processEvents()
                path = root if not suffix else root.with_name(root.stem + suffix + root.suffix)
                window.grab().save(str(path), "PNG")
            window.tabs.setCurrentIndex(0)
            app.processEvents()
        result = {
            "native_window_created": True, "qt6": True,
            "tabs": [window.tabs.tabText(i) for i in range(window.tabs.count())],
            "agents_max": 16, "corpus_partitions": [4, 16],
            "model_management": window.native_tree.columnCount() == 2,
            "mission_controller": not window.mission_writes.isChecked() and not window.mission_tests.isChecked(),
            "splitter_panels": window.main_splitter.count(),
            "browser_required": False,
        }
        Path(args.smoke_gui).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        window.close()
        return 0
    window = MainWindow(config, backend, provider=args.provider)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
