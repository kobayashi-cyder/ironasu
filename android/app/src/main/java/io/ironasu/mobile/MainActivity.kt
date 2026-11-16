package io.ironasu.mobile

import android.os.Bundle
import android.provider.OpenableColumns
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.runtime.snapshots.SnapshotStateList
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

private val palette = darkColorScheme(
    primary = androidx.compose.ui.graphics.Color(0xFFB0CEFF),
    secondary = androidx.compose.ui.graphics.Color(0xFF8FE4D1),
    background = androidx.compose.ui.graphics.Color(0xFF11131B),
    surface = androidx.compose.ui.graphics.Color(0xFF1B202B),
    surfaceVariant = androidx.compose.ui.graphics.Color(0xFF293142)
)

private data class Message(val role: String, val text: String)
private data class ModelEntry(val id: String, val label: String, val loaded: Boolean)
private data class CatalogEntry(val id: String, val downloads: Int)
private data class GGUFEntry(val path: String, val bytes: Long)
private val agentRoles = listOf(
    "仕様・要件", "実装方針", "安全性", "性能", "メモリ", "並列処理", "モデル適合性", "UX",
    "テスト", "障害解析", "依存関係", "ドキュメント", "データ品質", "代替案", "反証", "統合レビュー")

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme(colorScheme = palette) { IronasuApp() } }
    }
}

@Composable
private fun IronasuApp() {
    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("ironasu-public-settings", 0) }
    val scope = rememberCoroutineScope()
    val localEngine = remember { LocalEngine(context.applicationContext) }
    DisposableEffect(localEngine) { onDispose { localEngine.close() } }
    var page by remember { mutableIntStateOf(0) }
    var server by remember { mutableStateOf(prefs.getString("server", "http://192.168.1.100:8766") ?: "") }
    var apiKey by remember { mutableStateOf("") } // Deliberately never persisted.
    var client by remember { mutableStateOf<GatewayClient?>(null) }
    var online by remember { mutableStateOf(false) }
    var remoteInfo by remember { mutableStateOf(JSONObject()) }
    var error by remember { mutableStateOf("") }
    var connecting by remember { mutableStateOf(false) }
    var localMode by remember { mutableStateOf(false) }
    var localModelImported by remember { mutableStateOf(File(context.filesDir, "models/active.task").isFile) }
    var importing by remember { mutableStateOf(false) }
    var selectedModel by remember { mutableStateOf("ironasu") }
    var chatBusy by remember { mutableStateOf(false) }
    var chatPrompt by remember { mutableStateOf("") }
    val chat = remember { mutableStateListOf<Message>() }
    var agentsGoal by remember { mutableStateOf("") }
    var agentCount by remember { mutableIntStateOf(4) }
    var modelSlots by remember { mutableIntStateOf(2) }
    var agentBusy by remember { mutableStateOf(false) }
    var agentJob by remember { mutableStateOf(JSONObject()) }
    var agentError by remember { mutableStateOf("") }
    var provider by remember { mutableStateOf("native") }
    var installed by remember { mutableStateOf(listOf<ModelEntry>()) }
    var catalog by remember { mutableStateOf(listOf<CatalogEntry>()) }
    var files by remember { mutableStateOf(listOf<GGUFEntry>()) }
    var modelSearch by remember { mutableStateOf("") }
    var selectedRepo by remember { mutableStateOf("") }
    var remoteModelId by remember { mutableStateOf("") }
    var modelMessage by remember { mutableStateOf("") }
    var modelBusy by remember { mutableStateOf(false) }
    var downloads by remember { mutableStateOf("") }

    suspend fun refreshStatus(api: GatewayClient) {
        remoteInfo = api.get("/ui/status")
        online = true
    }
    suspend fun refreshInstalled(api: GatewayClient, p: String) {
        val raw = api.get("/ui/models/installed?provider=${GatewayClient.encode(p)}")
        installed = raw.optJSONArray("models").orEmptyObjects().map {
            ModelEntry(it.optString("id"), it.optString("label", it.optString("id")),
                it.optBoolean("loaded"))
        }
    }
    suspend fun refreshDownloads(api: GatewayClient) {
        val jobs = api.get("/ui/models/downloads").optJSONArray("jobs").orEmptyObjects()
        downloads = jobs.take(8).joinToString("\n") {
            "${it.optString("model", "download").take(64)}: ${it.optString("state")}" }
    }
    fun startAgentPolling(api: GatewayClient, id: String) {
        scope.launch {
            repeat(600) { // Poll for up to 30 minutes, not an infinite loop.
                delay(3000)
                try {
                    val response = api.get("/ui/agents/jobs/$id")
                    agentJob = response
                    if (response.optString("state") !in listOf("queued", "running")) {
                        agentBusy = false
                        return@launch
                    }
                } catch (exc: Exception) {
                    agentError = "状態を取得できません: ${exc.message}"
                    agentBusy = false
                    return@launch
                }
            }
            agentBusy = false
            agentError = "監視期限に達しました。再接続してジョブ一覧を確認してください。"
        }
    }
    val importer = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            importing = true
            scope.launch {
                try {
                    withContext(Dispatchers.IO) {
                        val displayName = context.contentResolver.query(uri,
                            arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { cursor ->
                            if (cursor.moveToFirst()) cursor.getString(0) else null
                        } ?: ""
                        require(displayName.endsWith(".task", ignoreCase = true)) {
                            "MediaPipe-compatible .task file is required; GGUF is not supported locally"
                        }
                        val folder = File(context.filesDir, "models")
                        folder.mkdirs()
                        val temp = File(folder, "active.task.part")
                        try {
                            context.contentResolver.openInputStream(uri)?.use { input ->
                                temp.outputStream().use { output ->
                                    val buf = ByteArray(1024 * 1024)
                                    var total = 0L
                                    while (true) {
                                        val n = input.read(buf)
                                        if (n < 0) break
                                        total += n
                                        require(total <= 8L * 1024 * 1024 * 1024) { "Model exceeds 8 GiB limit" }
                                        output.write(buf, 0, n)
                                    }
                                    require(total > 0L) { "Empty model" }
                                }
                            } ?: error("Model file could not be opened")
                            localEngine.close()
                            val destination = File(folder, "active.task")
                            require(temp.renameTo(destination)) { "Could not activate imported model" }
                        } finally { temp.delete() }
                    }
                    localModelImported = true
                    modelMessage = "端末に .task を保存しました。互換性は初回推論時に検証されます。"
                } catch (exc: Exception) { modelMessage = exc.message ?: "モデルの読み込みに失敗" }
                finally { importing = false }
            }
        }
    }

    Scaffold(bottomBar = {
        NavigationBar {
            listOf("会話", "エージェント", "モデル", "接続").forEachIndexed { index, label ->
                NavigationBarItem(selected = page == index, onClick = { page = index },
                    icon = { Text(listOf("◉", "◎", "▤", "⚙")[index]) }, label = { Text(label) })
            }
        }
    }) { insets ->
        Column(Modifier.fillMaxSize().padding(insets).padding(horizontal = 16.dp)) {
            Row(Modifier.fillMaxWidth().padding(vertical = 14.dp), verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text("ironasu", fontSize = 25.sp, fontWeight = FontWeight.Bold)
                    Text(if (localMode) "ON DEVICE / MediaPipe" else if (online) "PC CONNECTED" else "PC DISCONNECTED",
                        style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.secondary)
                }
                Text(if (online) "● PC" else "○ PC", style = MaterialTheme.typography.labelMedium,
                    color = if (online) MaterialTheme.colorScheme.secondary else MaterialTheme.colorScheme.onSurfaceVariant)
            }
            when (page) {
                0 -> ChatPage(chat, chatPrompt, { chatPrompt = it }, chatBusy,
                    localMode, { localMode = it }, localModelImported,
                    selectedModel, remoteInfo.optJSONArray("upstream_models").orEmptyStrings(),
                    onModel = { selectedModel = it }, onSend = {
                        val prompt = chatPrompt.trim()
                        if (prompt.isEmpty() || chatBusy) return@ChatPage
                        chatPrompt = ""
                        chat.add(Message("user", prompt))
                        chatBusy = true
                        scope.launch {
                            try {
                                val content = if (localMode) {
                                    val contextText = chat.takeLast(8).joinToString("\n") {
                                        "${if (it.role == "user") "User" else "Assistant"}: ${it.text}" }
                                    localEngine.generate(File(context.filesDir, "models/active.task"), contextText + "\nAssistant:")
                                } else {
                                    val api = client ?: error("接続画面でPCに接続してください")
                                    api.chat(chat.map { it.role to it.text }, selectedModel)
                                }
                                chat.add(Message("assistant", content))
                            } catch (exc: Exception) { chat.add(Message("error", exc.message ?: "推論失敗")) }
                            finally { chatBusy = false }
                        }
                    }, onClear = { chat.clear() })
                1 -> AgentsPage(agentsGoal, { agentsGoal = it }, agentCount, { agentCount = it },
                    modelSlots, { modelSlots = it }, agentBusy, agentJob, agentError, onStart = {
                        val api = client
                        if (api == null) { agentError = "PC接続が必要です"; return@AgentsPage }
                        val goal = agentsGoal.trim()
                        if (goal.isEmpty()) { agentError = "依頼内容を入力してください"; return@AgentsPage }
                        agentBusy = true
                        agentError = ""
                        scope.launch {
                            try {
                                val taskArray = JSONArray()
                                repeat(agentCount) { index ->
                                    taskArray.put(JSONObject().put("task_id", "agent_${(index+1).toString().padStart(2,'0')}")
                                        .put("prompt", "役割: ${agentRoles[index]}。独立して検討し、根拠・懸念・結論を示す。\n依頼: $goal"))
                                }
                                val response = api.post("/ui/agents/start", JSONObject()
                                    .put("workers", agentCount).put("model_slots", modelSlots)
                                    .put("tasks", taskArray))
                                agentJob = response
                                startAgentPolling(api, response.getString("id"))
                            } catch (exc: Exception) {
                                agentError = exc.message ?: "ジョブ投入失敗"; agentBusy = false
                            }
                        }
                    })
                2 -> ModelsPage(online, localModelImported, importing, { importer.launch(arrayOf("*/*")) },
                    provider, { p -> provider = p; installed = emptyList(); catalog = emptyList(); files = emptyList() },
                    installed, catalog, files, modelSearch, { modelSearch = it }, selectedRepo,
                    remoteModelId, { remoteModelId = it }, modelBusy, modelMessage, downloads,
                    onRefresh = {
                        val api = client ?: return@ModelsPage
                        modelBusy = true
                        scope.launch {
                            try { refreshInstalled(api, provider); refreshDownloads(api); modelMessage = "取得しました" }
                            catch (exc: Exception) { modelMessage = exc.message ?: "失敗" }
                            finally { modelBusy = false }
                        }
                    }, onSearch = {
                        val api = client ?: return@ModelsPage
                        modelBusy = true
                        scope.launch {
                            try {
                                val result = api.get("/ui/models/native/search?q=${GatewayClient.encode(modelSearch)}")
                                catalog = result.optJSONArray("models").orEmptyObjects().map {
                                    CatalogEntry(it.optString("id"), it.optInt("downloads")) }
                                modelMessage = "${catalog.size} 件"
                            } catch (exc: Exception) { modelMessage = exc.message ?: "検索失敗" }
                            finally { modelBusy = false }
                        }
                    }, onRepo = { repo ->
                        val api = client
                        if (api != null) {
                            selectedRepo = repo; files = emptyList(); modelBusy = true
                            scope.launch {
                                try {
                                    val result = api.get("/ui/models/native/files?repo=${GatewayClient.encode(repo)}")
                                    files = result.optJSONArray("files").orEmptyObjects().map {
                                        GGUFEntry(it.optString("path"), it.optLong("size_bytes")) }
                                } catch (exc: Exception) { modelMessage = exc.message ?: "ファイル一覧取得失敗" }
                                finally { modelBusy = false }
                            }
                        }
                    }, onDownloadFile = { file ->
                        val api = client
                        if (api != null) {
                            modelBusy = true
                            scope.launch {
                                try {
                                    api.post("/ui/models/download", JSONObject()
                                        .put("provider", "native").put("model", selectedRepo).put("file", file))
                                    modelMessage = "PC側のダウンロードを開始しました"
                                    refreshDownloads(api)
                                } catch (exc: Exception) { modelMessage = exc.message ?: "ダウンロード失敗" }
                                finally { modelBusy = false }
                            }
                        }
                    }, onDownloadExternal = {
                        val api = client
                        if (api != null) {
                            modelBusy = true
                            scope.launch {
                                try {
                                    api.post("/ui/models/download", JSONObject()
                                        .put("provider", provider).put("model", remoteModelId))
                                    modelMessage = "PC側のダウンロードを開始しました"; refreshDownloads(api)
                                } catch (exc: Exception) { modelMessage = exc.message ?: "ダウンロード失敗" }
                                finally { modelBusy = false }
                            }
                        }
                    }, onModelAction = { entry ->
                        val api = client
                        if (api != null) {
                            modelBusy = true
                            scope.launch {
                                try {
                                    if (provider == "native") {
                                        api.post("/ui/runtime/start", JSONObject().put("id", entry.id)
                                            .put("context_length", 4096))
                                        modelMessage = "PCのGGUFランタイム起動を要求しました"
                                    } else if (provider == "lmstudio") {
                                        api.post("/ui/models/load", JSONObject().put("provider", provider)
                                            .put("model", entry.id))
                                        modelMessage = "LM Studioのモデルロードを要求しました"
                                    } else modelMessage = "Ollamaは推論時にロードされます"
                                    refreshInstalled(api, provider)
                                } catch (exc: Exception) { modelMessage = exc.message ?: "ロード失敗" }
                                finally { modelBusy = false }
                            }
                        }
                    }, onRuntimeStop = {
                        val api = client
                        if (api != null) scope.launch {
                            try { api.post("/ui/runtime/stop", JSONObject()); modelMessage = "停止しました" }
                            catch (exc: Exception) { modelMessage = exc.message ?: "停止失敗" }
                        }
                    })
                3 -> SettingsPage(server, { server = it }, apiKey, { apiKey = it },
                    connecting, online, remoteInfo, error,
                    onConnect = {
                        connecting = true; error = ""
                        scope.launch {
                            try {
                                val api = GatewayClient(server, apiKey)
                                api.get("/health")
                                refreshStatus(api)
                                client = api
                                prefs.edit().putString("server", server.trim()).apply()
                                refreshInstalled(api, provider)
                            } catch (exc: Exception) {
                                online = false; client = null
                                error = exc.message ?: "接続に失敗しました"
                            } finally { connecting = false }
                        }
                    }, onDisconnect = { client = null; online = false; remoteInfo = JSONObject() })
            }
        }
    }
}

private fun JSONArray?.orEmptyObjects(): List<JSONObject> =
    if (this == null) emptyList() else (0 until length()).mapNotNull { optJSONObject(it) }
private fun JSONArray?.orEmptyStrings(): List<String> =
    if (this == null) emptyList() else (0 until length()).map { optString(it) }.filter { it.isNotBlank() }

@Composable
private fun Panel(title: String, content: @Composable ColumnScope.() -> Unit) {
    Card(Modifier.fillMaxWidth().padding(vertical = 6.dp), shape = RoundedCornerShape(18.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)) {
        Column(Modifier.fillMaxWidth().padding(16.dp), verticalArrangement = Arrangement.spacedBy(9.dp)) {
            Text(title, fontWeight = FontWeight.SemiBold, fontSize = 18.sp)
            content()
        }
    }
}

@Composable
private fun ChatPage(messages: SnapshotStateList<Message>, prompt: String, onPrompt: (String) -> Unit,
                     busy: Boolean, localMode: Boolean, onMode: (Boolean) -> Unit,
                     localReady: Boolean, selectedModel: String, modelOptions: List<String>,
                     onModel: (String) -> Unit, onSend: () -> Unit, onClear: () -> Unit) {
    Column(Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("端末内で推論", Modifier.weight(1f))
            Switch(checked = localMode, onCheckedChange = onMode)
        }
        if (localMode) {
            Text(if (localReady) "端末内 .task モデルを使用します（モデル互換性は端末に依存）。"
                 else "モデルタブで .task をインポートしてください。",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
        } else {
            var expanded by remember { mutableStateOf(false) }
            Box {
                OutlinedButton(onClick = { expanded = true }) { Text("モデル: ${selectedModel.take(35)}") }
                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    (listOf("ironasu", "ironasu-fast", "ironasu-quality") + modelOptions.map { "direct:$it" })
                        .distinct().forEach { model ->
                            DropdownMenuItem(text = { Text(model.take(65)) }, onClick = {
                                expanded = false; onModel(model)
                            })
                        }
                }
            }
        }
        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
            Text("会話履歴", fontWeight = FontWeight.Medium, modifier = Modifier.weight(1f))
            TextButton(onClick = onClear, enabled = !busy) { Text("クリア") }
        }
        Column(Modifier.weight(1f).fillMaxWidth().verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (messages.isEmpty()) Text("メッセージを入力して推論を開始します。",
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            messages.forEach { m ->
                Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(
                    containerColor = if (m.role == "user") MaterialTheme.colorScheme.surfaceVariant
                    else MaterialTheme.colorScheme.surface)) {
                    Column(Modifier.padding(12.dp)) {
                        Text(when (m.role) { "user" -> "YOU"; "error" -> "ERROR"; else -> "IRONASU" },
                            fontSize = 11.sp, fontWeight = FontWeight.Bold,
                            color = if (m.role == "error") MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.secondary)
                        Text(m.text, fontSize = 15.sp)
                    }
                }
            }
            if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())
        }
        OutlinedTextField(value = prompt, onValueChange = onPrompt, modifier = Modifier.fillMaxWidth(),
            label = { Text("メッセージ") }, minLines = 2, maxLines = 5)
        Button(onClick = onSend, enabled = !busy && prompt.isNotBlank() && (!localMode || localReady),
            modifier = Modifier.fillMaxWidth()) { Text(if (busy) "推論中..." else "送信") }
        Spacer(Modifier.height(3.dp))
    }
}

@Composable
private fun AgentsPage(goal: String, onGoal: (String) -> Unit, workers: Int, onWorkers: (Int) -> Unit,
                       slots: Int, onSlots: (Int) -> Unit, busy: Boolean, job: JSONObject,
                       error: String, onStart: () -> Unit) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        Panel("PC側エージェント管理") {
            Text("既存の読み取り専用ハーネスを使用。最大16エージェント。書込み・テスト実行は許可しません。",
                style = MaterialTheme.typography.bodySmall)
            OutlinedTextField(goal, onGoal, Modifier.fillMaxWidth(),
                label = { Text("チームに依頼する内容") }, minLines = 3, maxLines = 7)
            Text("エージェント数: $workers")
            Slider(workers.toFloat(), { onWorkers(it.toInt().coerceIn(1, 16)) },
                valueRange = 1f..16f, steps = 14)
            Text("同時LLM呼出し上限（希望）: $slots")
            Slider(slots.toFloat(), { onSlots(it.toInt().coerceIn(1, 16)) },
                valueRange = 1f..16f, steps = 14)
            Text("実際の同時推論数はPC側の IRONASU_MAX_CONCURRENCY 以下になります。",
                style = MaterialTheme.typography.bodySmall)
            Button(onClick = onStart, enabled = !busy && goal.isNotBlank(), modifier = Modifier.fillMaxWidth()) {
                Text(if (busy) "エージェント実行中" else "$workers エージェントで実行")
            }
            if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())
            if (error.isNotBlank()) Text(error, color = MaterialTheme.colorScheme.error)
        }
        if (job.has("id")) Panel("ジョブ結果") {
            Text("ID: ${job.optString("id")}")
            Text("状態: ${job.optString("state")}")
            Text("割当ワーカー: ${job.optInt("workers")}, LLM呼出し枠: ${job.optInt("model_slots")}")
            job.optString("error").takeIf { it.isNotBlank() }?.let {
                Text(it, color = MaterialTheme.colorScheme.error)
            }
            job.optJSONArray("results").orEmptyObjects().forEach { result ->
                HorizontalDivider()
                Text("${result.optString("task_id")}: ${result.optString("status")}",
                    fontWeight = FontWeight.SemiBold)
                Text(result.optString("text", "").take(6000))
                if (result.optString("error").isNotBlank())
                    Text(result.optString("error"), color = MaterialTheme.colorScheme.error)
            }
            if (job.optJSONArray("results").orEmptyObjects().isEmpty())
                Text("結果を待っています。モデル処理時間はPCの負荷に依存します。")
        }
    }
}

@Composable
private fun ModelsPage(online: Boolean, localReady: Boolean, importing: Boolean,
                       onImport: () -> Unit, provider: String, onProvider: (String) -> Unit,
                       installed: List<ModelEntry>, catalog: List<CatalogEntry>, files: List<GGUFEntry>,
                       query: String, onQuery: (String) -> Unit, repo: String,
                       remoteModelId: String, onRemoteModelId: (String) -> Unit,
                       busy: Boolean, info: String, downloads: String,
                       onRefresh: () -> Unit, onSearch: () -> Unit, onRepo: (String) -> Unit,
                       onDownloadFile: (String) -> Unit, onDownloadExternal: () -> Unit,
                       onModelAction: (ModelEntry) -> Unit, onRuntimeStop: () -> Unit) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        Panel("端末内モデル — オフライン") {
            Text("MediaPipe形式の .task を端末に取り込みます。PCのGGUFとは形式が異なります。",
                style = MaterialTheme.typography.bodySmall)
            Text(if (localReady) "保存済み: active.task" else "未導入")
            Button(onClick = onImport, enabled = !importing, modifier = Modifier.fillMaxWidth()) {
                Text(if (importing) "コピー中..." else ".task ファイルを選択して導入")
            }
        }
        Panel("PC側モデル管理") {
            Text(if (online) "PCと接続済み" else "接続画面でPCに接続してください。")
            var expanded by remember { mutableStateOf(false) }
            Box {
                OutlinedButton(onClick = { expanded = true }) { Text("保存先: $provider") }
                DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                    listOf("native", "lmstudio", "ollama").forEach { item ->
                        DropdownMenuItem(text = { Text(item) }, onClick = {
                            onProvider(item); expanded = false
                        })
                    }
                }
            }
            Button(onClick = onRefresh, enabled = online && !busy, modifier = Modifier.fillMaxWidth()) { Text("インストール済みを更新") }
            if (provider == "native") {
                OutlinedTextField(query, onQuery, Modifier.fillMaxWidth(),
                    label = { Text("Hugging Face公開モデルを検索") }, singleLine = true)
                OutlinedButton(onClick = onSearch, enabled = online && !busy && query.length in 2..100) {
                    Text("検索")
                }
                catalog.forEach { entry ->
                    TextButton(onClick = { onRepo(entry.id) }) {
                        Text("${entry.id} (${entry.downloads} downloads)")
                    }
                }
                if (repo.isNotBlank()) Text("選択中: $repo")
                files.take(100).forEach { entry ->
                    Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(entry.path, style = MaterialTheme.typography.bodySmall)
                            Text("${entry.bytes / 1024 / 1024} MiB", style = MaterialTheme.typography.labelSmall)
                        }
                        TextButton(onClick = { onDownloadFile(entry.path) }, enabled = online && !busy) {
                            Text("PCへ取得")
                        }
                    }
                }
            } else {
                OutlinedTextField(remoteModelId, onRemoteModelId, Modifier.fillMaxWidth(),
                    label = { Text("モデルID") }, singleLine = true)
                Button(onClick = onDownloadExternal, enabled = online && !busy && remoteModelId.isNotBlank()) {
                    Text("PCへダウンロード")
                }
            }
            if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())
            if (info.isNotBlank()) Text(info, style = MaterialTheme.typography.bodySmall)
            Text("インストール済み", fontWeight = FontWeight.SemiBold)
            if (installed.isEmpty()) Text("モデル一覧は未取得、または空です。", style = MaterialTheme.typography.bodySmall)
            installed.forEach { entry ->
                Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                    Text(entry.label.take(75) + if (entry.loaded) "  ●" else "",
                        Modifier.weight(1f), style = MaterialTheme.typography.bodySmall)
                    if (provider != "ollama") TextButton(onClick = { onModelAction(entry) }, enabled = !busy) {
                        Text(if (provider == "native") "起動" else "ロード")
                    }
                }
            }
            if (provider == "native") OutlinedButton(onClick = onRuntimeStop, enabled = online) {
                Text("PC管理GGUFを停止")
            }
            if (downloads.isNotBlank()) {
                Text("ダウンロード状況", fontWeight = FontWeight.SemiBold)
                Text(downloads, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

@Composable
private fun SettingsPage(server: String, onServer: (String) -> Unit,
                         token: String, onToken: (String) -> Unit, busy: Boolean,
                         connected: Boolean, info: JSONObject, error: String,
                         onConnect: () -> Unit, onDisconnect: () -> Unit) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        Panel("PC接続設定") {
            Text("PCの ironasu Studio をLAN/VPN経由で利用します。ブラウザは不要です。",
                style = MaterialTheme.typography.bodySmall)
            OutlinedTextField(server, onServer, Modifier.fillMaxWidth(),
                label = { Text("ironasu のURL") }, singleLine = true)
            OutlinedTextField(token, onToken, Modifier.fillMaxWidth(),
                label = { Text("IRONASU_SERVER_KEY（この起動中のみ保持）") }, singleLine = true,
                visualTransformation = androidx.compose.ui.text.input.PasswordVisualTransformation())
            Button(onClick = onConnect, enabled = !busy, modifier = Modifier.fillMaxWidth()) {
                Text(if (busy) "接続確認中..." else "接続・認証")
            }
            if (connected) OutlinedButton(onClick = onDisconnect, modifier = Modifier.fillMaxWidth()) { Text("切断") }
            if (error.isNotBlank()) Text(error, color = MaterialTheme.colorScheme.error)
            Text(if (connected) "認証済み" else "未接続")
        }
        if (connected) Panel("PC実行状況") {
            Text("プロバイダー: ${info.optString("provider", "不明")}")
            Text("高速モデル: ${info.optString("fast_model", "未検出")}")
            Text("高品質モデル: ${info.optString("quality_model", "未検出")}")
            Text("上流: ${if (info.optBoolean("upstream_available")) "利用可能" else "モデルサーバー未検出"}")
            val metrics = info.optJSONObject("metrics") ?: JSONObject()
            Text("実行中: ${metrics.optInt("active_requests")}, 待機中: ${metrics.optInt("waiting_requests")}")
            Text("キャッシュヒット: ${metrics.optInt("cache_hits")}")
        }
        Panel("ネットワークの注意") {
            Text("HTTP接続はプライベートネットワークアドレスに限定しています。ただし無線LAN上の平文HTTPではトークンと会話内容が盗聴され得ます。信頼できるLANまたはHTTPS/VPNを使用してください。",
                style = MaterialTheme.typography.bodySmall)
            Text("PC側で IRONASU_HOST=0.0.0.0 と IRONASU_SERVER_KEY を設定します。公開インターネットには直接開放しないでください。",
                style = MaterialTheme.typography.bodySmall)
        }
    }
}