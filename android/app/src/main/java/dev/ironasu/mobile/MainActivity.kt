package dev.ironasu.mobile

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.roundToInt

private val ink = Color(0xFF0A0E18)
private val surface = Color(0xFF182233)
private val blue = Color(0xFF9CB9FE)

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val ui = remember { UiState(LocalModel(applicationContext)) }
            val scope = rememberCoroutineScope()
            MaterialTheme(colorScheme = darkColorScheme(primary = blue, background = ink,
                surface = surface, onSurface = Color(0xFFE8EDF6),
                onBackground = Color(0xFFE8EDF6))) {
                IronasuShell(ui, scope)
            }
        }
    }
}

private class UiState(val local: LocalModel) {
    var page by mutableIntStateOf(0)
    var endpoint by mutableStateOf("")
    var key by mutableStateOf("")
    var online by mutableStateOf(false)
    var offline by mutableStateOf(false)
    var localReady by mutableStateOf(local.hasModel())
    var busy by mutableStateOf(false)
    var error by mutableStateOf("")
    var note by mutableStateOf("")
    var prompt by mutableStateOf("")
    val chat = mutableStateListOf<ChatLine>()
    var corpus by mutableStateOf("")
    var objective by mutableStateOf("重要事項、矛盾、改善案を根拠と共にまとめてください")
    var workers by mutableIntStateOf(4)
    var partitions by mutableIntStateOf(4)
    var slots by mutableIntStateOf(2)
    var harnessTask by mutableStateOf("")
    var jobId by mutableStateOf("")
    var job by mutableStateOf<JSONObject?>(null)
    var provider by mutableStateOf("lmstudio")
    var modelName by mutableStateOf("")
    var ggufFile by mutableStateOf("")
    var installed by mutableStateOf(JSONArray())
    var downloads by mutableStateOf(JSONArray())
    var search by mutableStateOf(JSONArray())
    var files by mutableStateOf(JSONArray())
    fun client() = IronasuClient(endpoint, key)
}

private fun task(scope: CoroutineScope, ui: UiState, body: suspend () -> Unit) {
    if (ui.busy) return
    scope.launch {
        ui.busy = true
        ui.error = ""
        ui.note = ""
        try { body() }
        catch (e: Exception) { ui.error = (e.localizedMessage ?: "不明なエラー").take(320) }
        finally { ui.busy = false }
    }
}

@Composable
private fun IronasuShell(ui: UiState, scope: CoroutineScope) {
    LaunchedEffect(ui.jobId, ui.endpoint) {
        val id = ui.jobId
        if (id.isBlank()) return@LaunchedEffect
        while (ui.jobId == id) {
            try {
                val result = ui.client().job(id)
                ui.job = result
                if (result.optString("state") !in listOf("queued", "running")) break
            } catch (e: Exception) {
                ui.error = "進捗取得に失敗: " + (e.localizedMessage ?: "")
                break
            }
            delay(1600)
        }
    }
    Scaffold(containerColor = ink, topBar = {
        Row(Modifier.fillMaxWidth().background(ink)
            .padding(horizontal = 20.dp, vertical = 16.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("ironasu", color = blue, fontWeight = FontWeight.Bold,
                    style = MaterialTheme.typography.headlineMedium)
                Text("Hybrid AI Studio", color = Color.LightGray,
                    style = MaterialTheme.typography.labelSmall)
            }
            Surface(shape = RoundedCornerShape(18.dp), color = Color(0xFF27374D)) {
                Text(if (ui.offline) "DEVICE" else if (ui.online) "PC ONLINE" else "OFFLINE",
                    modifier = Modifier.padding(9.dp), style = MaterialTheme.typography.labelSmall)
            }
        }
    }, bottomBar = {
        NavigationBar(containerColor = surface) {
            val labels = listOf("チャット", "エージェント", "モデル", "接続")
            val glyphs = listOf("◈", "✧", "⬡", "⌁")
            labels.forEachIndexed { index, label ->
                NavigationBarItem(selected = index == ui.page, onClick = { ui.page = index },
                    icon = { Text(glyphs[index]) }, label = { Text(label, maxLines = 1) })
            }
        }
    }) { insets ->
        Column(Modifier.fillMaxSize().padding(insets).padding(horizontal = 16.dp)) {
            if (ui.error.isNotEmpty()) Banner(ui.error, Color(0xFF54303C))
            if (ui.note.isNotEmpty()) Banner(ui.note, Color(0xFF26483F))
            when (ui.page) {
                0 -> ChatPage(ui, scope)
                1 -> AgentPage(ui, scope)
                2 -> ModelPage(ui, scope)
                else -> ConnectionPage(ui, scope)
            }
        }
    }
}

@Composable
private fun Banner(message: String, tone: Color) {
    Surface(modifier = Modifier.fillMaxWidth().padding(bottom = 8.dp),
        shape = RoundedCornerShape(12.dp), color = tone) {
        Text(message, Modifier.padding(12.dp), style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun Header(title: String, subtitle: String) {
    Spacer(Modifier.height(8.dp))
    Text(title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.Bold)
    Text(subtitle, color = Color.LightGray, style = MaterialTheme.typography.bodySmall)
    Spacer(Modifier.height(14.dp))
}

@Composable
private fun ChatPage(ui: UiState, scope: CoroutineScope) {
    Header("チャット", "PCの大型LLMと端末内推論を選択")
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        FilterChip(selected = !ui.offline, onClick = { ui.offline = false },
            label = { Text("PC") })
        FilterChip(selected = ui.offline, onClick = { ui.offline = true },
            label = { Text("端末内オフライン") })
    }
    LazyColumn(modifier = Modifier.weight(1f).fillMaxWidth(),
        contentPadding = PaddingValues(vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(9.dp)) {
        if (ui.chat.isEmpty()) {
            item { Banner("PC接続または端末内モデルを設定して開始してください。", surface) }
        }
        items(ui.chat.toList()) { line ->
            Surface(color = if (line.role == "user") Color(0xFF253C5D) else surface,
                shape = RoundedCornerShape(14.dp), modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp)) {
                    Text(if (line.role == "user") "YOU" else "IRONASU",
                        color = blue, style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold)
                    Spacer(Modifier.height(5.dp))
                    Text(line.text, style = MaterialTheme.typography.bodyMedium)
                }
            }
        }
    }
    if (ui.busy) LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
    OutlinedTextField(value = ui.prompt, onValueChange = { ui.prompt = it },
        modifier = Modifier.fillMaxWidth(), minLines = 2, maxLines = 5,
        label = { Text("メッセージを入力") })
    Spacer(Modifier.height(8.dp))
    Button(modifier = Modifier.fillMaxWidth(),
        enabled = !ui.busy && ui.prompt.isNotBlank() && (ui.offline || ui.online),
        onClick = {
            val input = ui.prompt.trim()
            ui.prompt = ""
            ui.chat.add(ChatLine("user", input))
            task(scope, ui) {
                val answer = if (ui.offline) {
                    val transcript = ui.chat.takeLast(6).joinToString("\n") {
                        (if (it.role == "user") "User: " else "Assistant: ") + it.text
                    }
                    ui.local.respond(transcript + "\nAssistant:")
                } else ui.client().chat(ui.chat)
                ui.chat.add(ChatLine("assistant", answer))
            }
        }) { Text("送信") }
    Spacer(Modifier.height(8.dp))
}

@Composable
private fun AgentPage(ui: UiState, scope: CoroutineScope) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        Header("エージェント ×16", "論理ワーカーと実際のモデル同時実行枠を分離")
        OutlinedTextField(ui.objective, { ui.objective = it },
            label = { Text("目的") }, modifier = Modifier.fillMaxWidth(), maxLines = 3)
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(ui.corpus, { ui.corpus = it },
            label = { Text("分析するコーパス") },
            modifier = Modifier.fillMaxWidth().heightIn(min = 120.dp), maxLines = 8)
        Spacer(Modifier.height(8.dp))
        Text("論理エージェント " + ui.workers + "/16")
        Slider(value = ui.workers.toFloat(), onValueChange = { ui.workers = it.roundToInt() },
            valueRange = 1f..16f, steps = 14)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(4, 16).forEach { n ->
                FilterChip(selected = ui.partitions == n, onClick = { ui.partitions = n },
                    label = { Text(n.toString() + "分割") })
            }
        }
        Text("モデル同時実行上限 " + ui.slots + "（PC設定値で自動調整）",
            style = MaterialTheme.typography.bodySmall)
        Slider(value = ui.slots.toFloat(), onValueChange = { ui.slots = it.roundToInt() },
            valueRange = 1f..16f, steps = 14)
        Button(modifier = Modifier.fillMaxWidth(),
            enabled = !ui.busy && ui.online && ui.corpus.length >= ui.partitions,
            onClick = {
                task(scope, ui) {
                    val job = ui.client().startAgents(ui.corpus, ui.objective,
                        ui.workers, ui.partitions, ui.slots)
                    ui.job = job
                    ui.jobId = job.getString("id")
                    ui.note = "分析を開始しました"
                }
            }) { Text("分析を開始") }
        Spacer(Modifier.height(14.dp))
        JobPanel(ui, scope)
        HorizontalDivider(Modifier.padding(vertical = 18.dp))
        Text("PCコードハーネス（読み取り専用）",
            style = MaterialTheme.typography.titleMedium)
        Text("PC側で明示的に許可したワークスペースだけにアクセスします。",
            color = Color.LightGray, style = MaterialTheme.typography.bodySmall)
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(ui.harnessTask, { ui.harnessTask = it },
            label = { Text("コード調査タスク") }, modifier = Modifier.fillMaxWidth(), maxLines = 4)
        Spacer(Modifier.height(8.dp))
        OutlinedButton(modifier = Modifier.fillMaxWidth(),
            enabled = !ui.busy && ui.online && ui.harnessTask.isNotBlank(),
            onClick = {
                task(scope, ui) {
                    val job = ui.client().startHarness(ui.harnessTask)
                    ui.job = job
                    ui.jobId = job.getString("id")
                    ui.note = "読み取り専用ハーネスを開始しました"
                }
            }) { Text("ハーネスを実行") }
        Spacer(Modifier.height(28.dp))
    }
}

@Composable
private fun JobPanel(ui: UiState, scope: CoroutineScope) {
    val job = ui.job ?: return
    Surface(color = surface, shape = RoundedCornerShape(16.dp),
        modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(14.dp)) {
            Text(job.optString("kind") + " · " + job.optString("state"),
                fontWeight = FontWeight.Bold)
            val done = job.optInt("completed")
            val total = job.optInt("total")
            Text("完了 " + done + "/" + if (total == 0) "未計測" else total.toString(),
                color = Color.LightGray, style = MaterialTheme.typography.bodySmall)
            if (total > 0) LinearProgressIndicator(
                progress = { (done.toFloat() / total).coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth())
            val result = job.optJSONObject("result")
            if (result != null) {
                val output = result.optString("synthesis")
                    .ifBlank { result.optString("output") }
                if (output.isNotBlank()) Text(output, Modifier.padding(vertical = 10.dp))
                val parts = result.optJSONArray("parts")
                if (parts != null) {
                    Text("モデル呼び出し " + result.optInt("model_calls") +
                        " / 重複再利用 " + result.optInt("cache_hits"),
                        color = blue, style = MaterialTheme.typography.labelSmall)
                    for (i in 0 until parts.length()) {
                        val part = parts.optJSONObject(i) ?: continue
                        Text("区画 " + (i + 1) + ": " + part.optString("status"),
                            fontWeight = FontWeight.Bold,
                            style = MaterialTheme.typography.labelMedium)
                        Text(part.optString("summary").ifBlank { part.optString("error") },
                            style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
            if (job.optString("error").isNotBlank()) {
                Text(job.optString("error"), color = Color(0xFFFFBEC9))
            }
            if (job.optString("state") in listOf("queued", "running")) {
                OutlinedButton(enabled = !ui.busy, onClick = {
                    task(scope, ui) { ui.job = ui.client().cancel(ui.jobId) }
                }) { Text("停止を要求") }
            }
        }
    }
}

@Composable
private fun ModelPage(ui: UiState, scope: CoroutineScope) {
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        Header("モデル管理", "LM Studio / Ollama / PC側GGUF")
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf("lmstudio", "ollama", "native").forEach { provider ->
                FilterChip(selected = ui.provider == provider,
                    onClick = { ui.provider = provider; ui.installed = JSONArray() },
                    label = { Text(when (provider) {
                        "lmstudio" -> "LM Studio"
                        "ollama" -> "Ollama"
                        else -> "GGUF"
                    }) })
            }
        }
        OutlinedButton(enabled = !ui.busy && ui.online,
            onClick = {
                task(scope, ui) {
                    ui.installed = ui.client().models(ui.provider)
                    ui.downloads = ui.client().downloads()
                }
            }) { Text("インストール済みを更新") }
        for (i in 0 until minOf(ui.installed.length(), 40)) {
            val item = ui.installed.optJSONObject(i) ?: continue
            val id = item.optString("id")
            Surface(color = surface, shape = RoundedCornerShape(12.dp),
                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                Column(Modifier.padding(12.dp)) {
                    Text(item.optString("label", id), fontWeight = FontWeight.Medium)
                    if (ui.provider != "ollama") {
                        TextButton(enabled = !ui.busy, onClick = {
                            task(scope, ui) {
                                if (ui.provider == "native") ui.client().startGguf(id)
                                else ui.client().loadLmStudio(id)
                                ui.note = "PC側へロード要求を送信しました"
                            }
                        }) { Text(if (ui.provider == "native") "PCでGGUF起動" else "LM Studioでロード") }
                    }
                }
            }
        }
        HorizontalDivider(Modifier.padding(vertical = 14.dp))
        Text("PCへモデルを追加", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(ui.modelName, { ui.modelName = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text(if (ui.provider == "native") "Hugging Faceリポジトリ" else "モデルID") })
        if (ui.provider == "native") {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(enabled = ui.online && !ui.busy, onClick = {
                    task(scope, ui) { ui.search = ui.client().searchGguf(ui.modelName) }
                }) { Text("検索") }
                OutlinedButton(enabled = ui.online && !ui.busy, onClick = {
                    task(scope, ui) { ui.files = ui.client().ggufFiles(ui.modelName) }
                }) { Text("ファイル一覧") }
            }
            for (i in 0 until minOf(ui.search.length(), 10)) {
                val model = ui.search.optJSONObject(i) ?: continue
                TextButton(onClick = { ui.modelName = model.optString("id") }) {
                    Text(model.optString("id")) }
            }
            for (i in 0 until minOf(ui.files.length(), 12)) {
                val file = ui.files.optJSONObject(i) ?: continue
                TextButton(onClick = { ui.ggufFile = file.optString("path") }) {
                    Text(file.optString("path")) }
            }
            OutlinedTextField(ui.ggufFile, { ui.ggufFile = it },
                label = { Text("GGUFファイル名") }, modifier = Modifier.fillMaxWidth())
        }
        Spacer(Modifier.height(8.dp))
        Button(modifier = Modifier.fillMaxWidth(),
            enabled = ui.online && !ui.busy && ui.modelName.isNotBlank() &&
                (ui.provider != "native" || ui.ggufFile.isNotBlank()),
            onClick = {
                task(scope, ui) {
                    ui.client().download(ui.provider, ui.modelName, ui.ggufFile)
                    ui.downloads = ui.client().downloads()
                    ui.note = "PCでダウンロードを開始しました"
                }
            }) { Text("PCにダウンロード") }
        Spacer(Modifier.height(14.dp))
        Text("ダウンロード状況", style = MaterialTheme.typography.titleMedium)
        OutlinedButton(enabled = ui.online && !ui.busy,
            onClick = { task(scope, ui) { ui.downloads = ui.client().downloads() } }) {
            Text("進捗を更新")
        }
        for (i in 0 until ui.downloads.length()) {
            val job = ui.downloads.optJSONObject(i) ?: continue
            Surface(color = surface, shape = RoundedCornerShape(12.dp),
                modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                Column(Modifier.padding(12.dp)) {
                    Text(job.optString("model") + " · " + job.optString("state"))
                    Text(job.optString("detail"),
                        color = Color.LightGray, style = MaterialTheme.typography.bodySmall)
                    val size = job.optLong("total_bytes")
                    if (size > 0) LinearProgressIndicator(
                        progress = { (job.optLong("completed_bytes").toFloat() / size).coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth())
                }
            }
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun ConnectionPage(ui: UiState, scope: CoroutineScope) {
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) task(scope, ui) {
            ui.local.importAndLoad(uri)
            ui.localReady = true
            ui.note = "端末内モデルを読み込みました"
        }
    }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
        Header("接続設定", "PCへの認証接続と端末単独モード")
        Text("PCのironasu", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(value = ui.endpoint,
            onValueChange = { ui.endpoint = it; ui.online = false },
            label = { Text("PCのアドレス") }, placeholder = { Text("http://192.168.1.20:8765") },
            modifier = Modifier.fillMaxWidth(), singleLine = true)
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(value = ui.key,
            onValueChange = { ui.key = it; ui.online = false },
            label = { Text("IRONASU_SERVER_KEY（保存しません）") },
            modifier = Modifier.fillMaxWidth(), singleLine = true)
        Spacer(Modifier.height(10.dp))
        Button(modifier = Modifier.fillMaxWidth(),
            enabled = !ui.busy && ui.endpoint.isNotBlank(),
            onClick = {
                task(scope, ui) {
                    val client = ui.client()
                    client.get("/health")
                    val status = client.status()
                    ui.online = true
                    ui.note = "接続しました · " + status.optString("provider")
                }
            }) { Text("接続を確認") }
        Spacer(Modifier.height(8.dp))
        Text("LANのHTTP通信は暗号化されません。信頼できるネットワークだけで使用し、外部接続にはHTTPSを使用してください。",
            style = MaterialTheme.typography.bodySmall, color = Color(0xFFFFC78D))
        HorizontalDivider(Modifier.padding(vertical = 20.dp))
        Text("端末内AI", style = MaterialTheme.typography.titleMedium)
        Text(if (ui.localReady) "モデルが保存されています" else "モデルは未登録です",
            color = if (ui.localReady) Color(0xFF9BDCC4) else Color.LightGray)
        Spacer(Modifier.height(8.dp))
        Text("対応するMediaPipe形式の .task モデルを取り込みます。PC側のGGUFとは別形式です。",
            style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
        Spacer(Modifier.height(10.dp))
        OutlinedButton(modifier = Modifier.fillMaxWidth(), enabled = !ui.busy,
            onClick = { picker.launch(arrayOf("application/octet-stream", "*/*")) }) {
            Text("モデルファイルを選択")
        }
        if (ui.localReady) {
            OutlinedButton(modifier = Modifier.fillMaxWidth(), enabled = !ui.busy,
                onClick = {
                    task(scope, ui) {
                        ui.local.loadExisting()
                        ui.note = "端末内モデルをロードしました"
                    }
                }) { Text("保存済みモデルをロード") }
        }
        Spacer(Modifier.height(10.dp))
        Text("オフライン推論には端末のRAMと保存領域が必要です。実機での動作確認が必要です。",
            style = MaterialTheme.typography.bodySmall, color = Color.LightGray)
        Spacer(Modifier.height(28.dp))
    }
}
