/* ironasu Studio: offline, dependency-free, same-origin client. */
"use strict";
(function () {
  const byId = (id) => document.getElementById(id);
  const ui = {
    sidebar: byId("sidebar"), scrim: byId("scrim"), inspector: byId("inspector"),
    sessionList: byId("session-list"), title: byId("chat-title"),
    welcome: byId("welcome"), messages: byId("messages"),
    prompt: byId("prompt"), form: byId("chat-form"),
    send: byId("send-button"), stop: byId("stop-button"),
    model: byId("model-select"), persist: byId("persist-history"),
    error: byId("error-banner"), meta: byId("completion-meta"),
    temp: byId("temperature"), topP: byId("top-p"),
    maxTokens: byId("max-tokens"), system: byId("system-prompt"),
    stream: byId("stream-toggle"), keyOverlay: byId("key-overlay"),
    keyForm: byId("key-form"), keyInput: byId("access-key"),
    keyError: byId("key-error")
  };
  const storageKey = "ironasu-studio-history-v1";
  const optKey = "ironasu-studio-save-enabled-v1";
  let sessions = [];
  let activeId = null;
  let bearer = "";
  let controller = null;
  let busy = false;
  let connected = false;
  const nowId = () => "s-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);

  function current() { return sessions.find((x) => x.id === activeId); }
  function persist() {
    try {
      if (ui.persist.checked) {
        localStorage.setItem(optKey, "yes");
        localStorage.setItem(storageKey, JSON.stringify({ sessions, activeId }));
      } else {
        localStorage.removeItem(optKey);
        localStorage.removeItem(storageKey);
      }
    } catch (_ignored) {
      showError("ブラウザの保存領域に書き込めません。履歴はこのタブ内に保持します。");
    }
  }
  function load() {
    try {
      if (localStorage.getItem(optKey) !== "yes") return;
      ui.persist.checked = true;
      const raw = JSON.parse(localStorage.getItem(storageKey) || "{}");
      if (Array.isArray(raw.sessions)) {
        sessions = raw.sessions.slice(-40).filter((x) =>
          x && typeof x.id === "string" && typeof x.title === "string" &&
          Array.isArray(x.messages) && x.messages.length <= 500 &&
          x.messages.every((m) => m && ["user", "assistant"].includes(m.role) &&
            typeof m.content === "string")
        );
        if (sessions.some((x) => x.id === raw.activeId)) activeId = raw.activeId;
      }
    } catch (_ignored) { sessions = []; activeId = null; }
  }
  function newSession() {
    const session = { id: nowId(), title: "新しいチャット", messages: [] };
    sessions.unshift(session);
    activeId = session.id;
    persist();
    render();
    ui.prompt.focus();
    closePanels();
  }
  function closePanels() {
    ui.sidebar.classList.remove("is-open");
    ui.inspector.classList.remove("is-open");
    ui.scrim.classList.remove("is-open");
    ui.scrim.hidden = true;
  }
  function showError(msg) {
    ui.error.textContent = msg;
    ui.error.hidden = false;
  }
  function clearError() { ui.error.hidden = true; ui.error.textContent = ""; }
  function textNode(tag, className, content) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    el.textContent = content;
    return el;
  }
  function renderSessions() {
    ui.sessionList.replaceChildren();
    sessions.forEach((session) => {
      const item = textNode("div", "session" + (session.id === activeId ? " active" : ""), "");
      const label = textNode("button", "session-name", "▤  " + session.title);
      label.type = "button";
      label.title = session.title;
      label.addEventListener("click", () => {
        if (busy || controller) return;
        activeId = session.id;
        persist();
        render();
        closePanels();
      });
      const del = textNode("button", "session-delete", "×");
      del.type = "button";
      del.title = "セッションを削除";
      del.setAttribute("aria-label", session.title + " を削除");
      del.addEventListener("click", () => {
        if (busy || controller) return;
        sessions = sessions.filter((x) => x.id !== session.id);
        if (!sessions.length) {
          newSession();
        } else {
          if (activeId === session.id) activeId = sessions[0].id;
          persist();
          render();
        }
      });
      item.append(label, del);
      ui.sessionList.append(item);
    });
  }
  function renderMessage(message, index) {
    const item = textNode("article", "message " + message.role, "");
    const avatar = textNode("div", "avatar", message.role === "user" ? "U" : "i");
    avatar.setAttribute("aria-hidden", "true");
    const body = textNode("div", "message-body", "");
    const heading = textNode("div", "message-head", message.role === "user" ? "あなた" : "ironasu");
    if (message.note) heading.append(textNode("small", "", message.note));
    const value = textNode("div", "message-content", message.content || (message.role === "assistant" ? "生成中…" : ""));
    body.append(heading, value);
    if (message.role === "assistant" && message.content) {
      const actions = textNode("div", "message-actions", "");
      const copy = textNode("button", "small-button", "コピー");
      copy.type = "button";
      copy.addEventListener("click", () => {
        navigator.clipboard.writeText(message.content).then(() => {
          copy.textContent = "コピー済み";
        }).catch(() => showError("クリップボードにコピーできませんでした。"));
      });
      actions.append(copy);
      body.append(actions);
    }
    item.append(avatar, body);
    item.dataset.messageIndex = String(index);
    return item;
  }
  function render() {
    const session = current();
    if (!session) return;
    ui.title.textContent = session.title;
    ui.welcome.hidden = session.messages.length !== 0;
    ui.messages.replaceChildren();
    session.messages.forEach((m, i) => ui.messages.append(renderMessage(m, i)));
    renderSessions();
    scrollBottom();
  }
  function scrollBottom() {
    const container = ui.messages.parentElement;
    container.scrollTop = container.scrollHeight;
  }
  function paintLast(message) {
    const items = ui.messages.querySelectorAll(".message");
    const el = items[items.length - 1];
    if (!el || !message) return;
    const content = el.querySelector(".message-content");
    const head = el.querySelector(".message-head");
    if (content) content.textContent = message.content || "生成中…";
    const note = head.querySelector("small");
    if (note) note.textContent = message.note || "";
    else if (message.note) head.append(textNode("small", "", message.note));
    scrollBottom();
  }
  function authHeaders() {
    const result = { "Content-Type": "application/json" };
    if (bearer) result.Authorization = "Bearer " + bearer;
    return result;
  }
  function askForKey() {
    ui.keyOverlay.hidden = false;
    ui.keyError.textContent = "";
    ui.keyInput.value = "";
    ui.keyInput.focus();
  }
  async function fetchJson(path, options) {
    const response = await fetch(path, {
      ...options, headers: { ...authHeaders(), ...(options && options.headers || {}) },
      cache: "no-store"
    });
    if (response.status === 401) { askForKey(); throw new Error("認証が必要です。"); }
    if (!response.ok) {
      let message = "HTTP " + response.status;
      try { const json = await response.json(); message = json.error?.message || message; } catch (_) { /* ignore */ }
      throw new Error(message);
    }
    return response.json();
  }
  function setConnection(ok, label) {
    connected = ok;
    const pill = byId("connection-status");
    pill.classList.toggle("online", ok);
    pill.classList.toggle("offline", !ok);
    byId("connection-text").textContent = label;
    pill.title = label;
  }
  async function refreshStatus() {
    try {
      const data = await fetchJson("/ui/status");
      setConnection(Boolean(data.upstream_available), data.upstream_available ? "接続: " + (data.provider || "custom") : "モデルサーバー未検出");
      byId("backend-provider").textContent = data.provider || "未検出";
      byId("backend-url").textContent = data.backend_url || "—";
      byId("fast-model").textContent = data.fast_model || "—";
      byId("quality-model").textContent = data.quality_model || "—";
      const m = data.metrics || {};
      const sumCalls = Object.values(m.backend_calls || {}).reduce((a, b) => a + (Number(b) || 0), 0);
      byId("stat-active").textContent = String(m.active_requests ?? "—");
      byId("stat-waiting").textContent = String(m.waiting_requests ?? "—");
      byId("stat-cache").textContent = String(m.cache_hits ?? "—");
      byId("stat-calls").textContent = String(sumCalls);
      byId("pressure-status").textContent = m.pressure ? "高負荷" : "通常";
      const modelList = byId("backend-models");
      modelList.replaceChildren();
      const names = Array.isArray(data.upstream_models) ? data.upstream_models : [];
      byId("model-count").textContent = String(names.length);
      if (!names.length) modelList.append(textNode("span", "muted", "上流モデルは取得できませんでした。"));
      names.slice(0, 80).forEach((id) => modelList.append(textNode("div", "model-item", id)));
      const selected = ui.model.value;
      ui.model.querySelectorAll("option[data-dynamic]").forEach((o) => o.remove());
      names.forEach((id) => {
        const opt = new Option(id, "direct:" + id);
        opt.dataset.dynamic = "1";
        ui.model.add(opt);
      });
      if (Array.from(ui.model.options).some((x) => x.value === selected)) ui.model.value = selected;
      else ui.model.value = "auto";
    } catch (error) {
      setConnection(false, "バックエンド確認不可");
      if (!ui.keyOverlay.hidden) return;
      byId("backend-models").replaceChildren(textNode("span", "muted", String(error.message)));
    }
  }
  function configPayload(session) {
    const messages = [];
    const system = ui.system.value.trim();
    if (system) messages.push({ role: "system", content: system });
    // Empty pending assistant messages and notes are never sent to inference.
    session.messages.forEach((m) => {
      if (m.content && m.note !== "エラー" && m.note !== "停止" && (m.role === "user" || m.role === "assistant")) {
        messages.push({ role: m.role, content: m.content });
      }
    });
    const payload = {
      messages, model: "ironasu", temperature: Number(ui.temp.value),
      top_p: Number(ui.topP.value), max_tokens: Number(ui.maxTokens.value),
      stream: ui.stream.checked, x_ironasu_cache: true
    };
    if (ui.model.value.startsWith("direct:")) {
      payload.x_ironasu_model = ui.model.value.slice(7);
    } else {
      payload.x_ironasu_mode = ui.model.value;
    }
    return payload;
  }
  function extractContent(choice) {
    if (!choice) return "";
    const delta = choice.delta || choice.message || {};
    if (typeof delta.content === "string") return delta.content;
    if (Array.isArray(delta.content)) return delta.content.filter((part) =>
      part && part.type === "text" && typeof part.text === "string"
    ).map((part) => part.text).join("");
    return "";
  }
  async function streamCompletion(payload, answer, signal) {
    const response = await fetch("/v1/chat/completions", {
      method: "POST", headers: authHeaders(), body: JSON.stringify(payload), signal, cache: "no-store"
    });
    if (response.status === 401) { askForKey(); throw new Error("認証が必要です。"); }
    if (!response.ok) {
      let message = "HTTP " + response.status;
      try { const err = await response.json(); message = err.error?.message || message; } catch (_) { /* ignore */ }
      throw new Error(message);
    }
    if (!response.body || !response.headers.get("content-type")?.includes("text/event-stream"))
      throw new Error("バックエンドはSSE形式の応答を返しませんでした。");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let sawDone = false;
    function consume(event) {
      const payloadLine = event.split(/\r?\n/).filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart()).join("\n");
      if (!payloadLine) return;
      if (payloadLine.trim() === "[DONE]") { sawDone = true; return; }
      let chunk;
      try { chunk = JSON.parse(payloadLine); } catch (_) { return; }
      if (chunk.error) throw new Error(chunk.error.message || "ストリーミングエラー");
      for (const choice of (chunk.choices || [])) {
        answer.content += extractContent(choice);
      }
      if (chunk.usage && typeof chunk.usage.completion_tokens === "number") {
        answer.tokens = chunk.usage.completion_tokens;
      }
      paintLast(answer);
    }
    try {
      while (true) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true }).replace(/\r\n/g, "\n");
        if (buffer.length > 1024 * 1024) throw new Error("SSEイベントが大きすぎます。");
        let index;
        while ((index = buffer.indexOf("\n\n")) !== -1) {
          consume(buffer.slice(0, index));
          buffer = buffer.slice(index + 2);
        }
      }
      buffer += decoder.decode();
      if (buffer.trim()) consume(buffer);
      if (!sawDone && !signal.aborted) {
        // Some SSE servers do not send [DONE]; a clean close is tolerated.
      }
    } finally {
      reader.releaseLock();
    }
  }
  async function submit() {
    if (busy) return;
    const prompt = ui.prompt.value.trim();
    if (!prompt) return;
    clearError();
    const session = current();
    if (!session) return;
    session.messages.push({ role: "user", content: prompt });
    if (session.title === "新しいチャット") session.title = prompt.replace(/\s+/g, " ").slice(0, 38);
    ui.prompt.value = "";
    const answer = { role: "assistant", content: "", note: "応答中" };
    session.messages.push(answer);
    persist();
    render();
    ui.welcome.hidden = true;
    busy = true;
    ui.send.hidden = true;
    ui.stop.hidden = false;
    ui.model.disabled = true;
    controller = new AbortController();
    const began = performance.now();
    try {
      const payload = configPayload(session);
      if (payload.stream) {
        await streamCompletion(payload, answer, controller.signal);
      } else {
        const data = await fetchJson("/v1/chat/completions", {
          method: "POST", body: JSON.stringify(payload), signal: controller.signal
        });
        answer.content = extractContent(data.choices?.[0]);
        answer.tokens = data.usage?.completion_tokens;
        const meta = data._ironasu || {};
        answer.note = (meta.route || "upstream") + " · " + (meta.source || "upstream");
      }
      const seconds = (performance.now() - began) / 1000;
      answer.note = (answer.note === "応答中" ? "完了" : answer.note) + " · " + seconds.toFixed(1) + "秒";
      if (typeof answer.tokens === "number" && seconds > 0.1) {
        answer.note += " · " + (answer.tokens / seconds).toFixed(1) + " tok/s (端末計測)";
      }
      if (!answer.content) answer.content = "(テキスト応答はありませんでした)";
      ui.meta.textContent = answer.note;
    } catch (error) {
      if (error.name === "AbortError") {
        answer.note = "停止";
        if (!answer.content) answer.content = "(生成を停止しました)";
      } else {
        answer.note = "エラー";
        // Failed generations must never be sent as conversation context.
        if (!answer.content) session.messages.pop();
        showError(error.message || "生成に失敗しました。");
      }
    } finally {
      busy = false;
      controller = null;
      ui.send.hidden = false;
      ui.stop.hidden = true;
      ui.model.disabled = false;
      persist();
      render();
      refreshStatus();
      ui.prompt.focus();
    }
  }

  byId("new-chat").addEventListener("click", () => { if (!busy) newSession(); });
  ui.form.addEventListener("submit", (e) => { e.preventDefault(); submit(); });
  ui.prompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault(); submit();
    }
  });
  ui.stop.addEventListener("click", () => { if (controller) controller.abort(); });
  ui.persist.addEventListener("change", persist);
  byId("export-chat").addEventListener("click", () => {
    const session = current();
    if (!session) return;
    const data = JSON.stringify({ title: session.title, messages: session.messages }, null, 2);
    const blob = new Blob([data], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url; link.download = "ironasu-chat.json";
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  document.querySelectorAll("[data-prompt]").forEach((btn) => btn.addEventListener("click", () => {
    ui.prompt.value = btn.dataset.prompt || "";
    ui.prompt.focus();
  }));
  [[ui.temp, byId("temperature-label")], [ui.topP, byId("top-p-label")]].forEach(([input, output]) => {
    input.addEventListener("input", () => { output.textContent = Number(input.value).toFixed(2); });
  });
  byId("connection-status").addEventListener("click", refreshStatus);
  byId("refresh-status").addEventListener("click", refreshStatus);
  byId("open-menu").addEventListener("click", () => {
    closePanels(); ui.sidebar.classList.add("is-open");
    ui.scrim.hidden = false; ui.scrim.classList.add("is-open");
  });
  byId("close-menu").addEventListener("click", closePanels);
  byId("toggle-inspector").addEventListener("click", () => {
    const open = ui.inspector.classList.contains("is-open");
    closePanels();
    if (!open) {
      ui.inspector.classList.add("is-open");
      ui.scrim.hidden = false; ui.scrim.classList.add("is-open");
    }
  });
  ui.scrim.addEventListener("click", closePanels);
  ui.keyForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    bearer = ui.keyInput.value;
    ui.keyInput.value = "";
    ui.keyOverlay.hidden = true;
    try {
      await fetchJson("/ui/status");
      await refreshStatus();
      ui.prompt.focus();
    } catch (error) {
      ui.keyOverlay.hidden = false;
      ui.keyError.textContent = error.message || "アクセスキーを確認してください。";
      ui.keyInput.focus();
    }
  });
  // Model management. All network calls target this same-origin gateway,
  // and provider selection is limited server-side to loopback LM Studio/Ollama.
  const modelDialog = byId("model-dialog");
  const installProvider = byId("install-provider");
  const installName = byId("install-model");
  const installQuant = byId("install-quantization");
  const installError = byId("model-install-error");
  let lastCompleted = "";
  let modelRefreshBusy = false;

  function updateModelExamples() {
    const isLM = installProvider.value === "lmstudio";
    installName.placeholder = isLM ? "ibm/granite-4-micro" : "qwen3:4b";
    installQuant.disabled = !isLM;
    if (!isLM) installQuant.value = "";
    const samples = isLM
      ? ["ibm/granite-4-micro", "openai/gpt-oss-20b"]
      : ["qwen3:4b", "gemma3:4b"];
    const examples = byId("install-examples");
    examples.replaceChildren();
    samples.forEach((value) => {
      const button = textNode("button", "small-button", value);
      button.type = "button";
      button.addEventListener("click", () => { installName.value = value; installName.focus(); });
      examples.append(button);
    });
  }

  function formatBytes(value) {
    if (typeof value !== "number" || value < 0) return "容量未取得";
    if (value === 0) return "0 B";
    const u = ["B", "KiB", "MiB", "GiB", "TiB"];
    const ix = Math.min(Math.floor(Math.log(value) / Math.log(1024)), u.length - 1);
    return (value / Math.pow(1024, ix)).toFixed(ix > 1 ? 1 : 0) + " " + u[ix];
  }

  async function refreshInstalledModels() {
    const target = byId("installed-model-list");
    try {
      const provider = installProvider.value;
      const data = await fetchJson("/ui/models/installed?provider=" + encodeURIComponent(provider));
      if (installProvider.value !== provider) return;
      target.replaceChildren();
      if (!Array.isArray(data.models) || data.models.length === 0) {
        target.append(textNode("span", "muted", "モデルはまだインストールされていません。"));
        return;
      }
      data.models.forEach((entry) => {
        const row = textNode("div", "model-result", "");
        const info = textNode("div", "model-result-text", "");
        info.append(textNode("strong", "", entry.label || entry.id));
        const detail = (entry.id || "") + " · " + formatBytes(entry.size_bytes);
        info.append(textNode("small", "", detail + (entry.loaded === true ? " · ロード済み" : "")));
        row.append(info);
        if (provider === "lmstudio" && entry.loaded === false) {
          const load = textNode("button", "small-button", "ロード");
          load.type = "button";
          load.addEventListener("click", async () => {
            load.disabled = true;
            installError.textContent = "";
            try {
              const loaded = await fetchJson("/ui/models/load", {
                method: "POST", body: JSON.stringify({ provider, model: entry.id })
              });
              if (loaded.status !== "loaded") throw new Error("モデルのロード結果を確認できません: " + loaded.status);
              await Promise.all([refreshStatus(), refreshInstalledModels()]);
            } catch (error) {
              installError.textContent = error.message;
            } finally {
              load.disabled = false;
            }
          });
          row.append(load);
        }
        target.append(row);
      });
    } catch (error) {
      target.replaceChildren(textNode("span", "muted", "取得失敗: " + error.message));
    }
  }

  async function refreshModelDownloads() {
    if (modelRefreshBusy) return;
    modelRefreshBusy = true;
    try {
      const data = await fetchJson("/ui/models/downloads");
      const target = byId("model-download-list");
      target.replaceChildren();
      const jobs = Array.isArray(data.jobs) ? data.jobs.slice(0, 30) : [];
      if (!jobs.length) target.append(textNode("span", "muted", "ダウンロード履歴なし"));
      jobs.forEach((job) => {
        const card = textNode("div", "download-job", "");
        card.append(textNode("strong", "", job.model + " · " + job.provider));
        card.append(textNode("small", "", (job.detail || job.state) + " [" + job.state + "]"));
        if (job.state === "downloading" || job.state === "pending") {
          const bar = document.createElement("progress");
          bar.max = 100;
          if (typeof job.total_bytes === "number" && job.total_bytes > 0
              && typeof job.completed_bytes === "number") {
            bar.value = Math.max(0, Math.min(100, 100 * job.completed_bytes / job.total_bytes));
          }
          card.append(bar);
          if (typeof job.completed_bytes === "number") {
            card.append(textNode("small", "", formatBytes(job.completed_bytes)
                + " / " + formatBytes(job.total_bytes)
                + (job.provider === "ollama" ? "（現在のレイヤー）" : "")));
          }
        }
        target.append(card);
      });
      const completed = jobs.filter((j) => j.state === "completed").map((j) => j.id).join("|");
      if (completed !== lastCompleted) {
        lastCompleted = completed;
        if (completed) {
          await refreshInstalledModels();
          await refreshStatus();
        }
      }
    } catch (error) {
      byId("model-download-list").replaceChildren(textNode("span", "muted", error.message));
    } finally {
      modelRefreshBusy = false;
    }
  }

  function closeModelDialog() {
    modelDialog.hidden = true;
    // Do not interrupt a download: it is owned by the local model server.
  }
  byId("open-model-manager").addEventListener("click", async () => {
    closePanels();
    modelDialog.hidden = false;
    installError.textContent = "";
    updateModelExamples();
    await Promise.all([refreshModelDownloads(), refreshInstalledModels()]);
    installName.focus();
  });
  byId("close-model-manager").addEventListener("click", closeModelDialog);
  modelDialog.addEventListener("click", (e) => {
    if (e.target === modelDialog) closeModelDialog();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modelDialog.hidden) closeModelDialog();
  });
  installProvider.addEventListener("change", () => {
    updateModelExamples();
    refreshInstalledModels();
  });
  byId("refresh-downloads").addEventListener("click", refreshModelDownloads);
  byId("refresh-installed").addEventListener("click", refreshInstalledModels);
  byId("model-install-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    installError.textContent = "";
    const button = byId("install-submit");
    button.disabled = true;
    try {
      const payload = {
        provider: installProvider.value,
        model: installName.value.trim()
      };
      if (installQuant.value.trim()) payload.quantization = installQuant.value.trim();
      const data = await fetchJson("/ui/models/download", {
        method: "POST", body: JSON.stringify(payload)
      });
      if (!data.job?.id) throw new Error("ダウンロードジョブが開始できませんでした。");
      await refreshModelDownloads();
    } catch (error) {
      installError.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });
  setInterval(() => {
    if (!modelDialog.hidden && !document.hidden) refreshModelDownloads();
  }, 1800);

  load();
  if (!sessions.length) newSession();
  else { if (!activeId) activeId = sessions[0].id; render(); }
  refreshStatus();
  setInterval(() => { if (!document.hidden && !busy && ui.keyOverlay.hidden) refreshStatus(); }, 10000);
})();
