package dev.ironasu.mobile

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.net.HttpURLConnection
import java.net.URI
import java.net.URL
import java.net.URLEncoder

/** Only explicit HTTPS endpoints or trusted private-IP HTTP hosts are accepted. */
object EndpointRules {
    fun normalize(input: String): String {
        val source = input.trim().trimEnd('/')
        val uri = try { URI(source) } catch (_: Exception) { throw IllegalArgumentException("URLが不正です") }
        val scheme = uri.scheme?.lowercase()
        val host = uri.host?.lowercase() ?: throw IllegalArgumentException("ホスト名が必要です")
        require(uri.userInfo == null && uri.query == null && uri.fragment == null && (uri.path.isNullOrEmpty() || uri.path == "/")) {
            "サーバーURLはホストとポートのみ指定してください"
        }
        require(scheme == "https" || scheme == "http") { "HTTPSまたはHTTPが必要です" }
        if (scheme == "http") {
            val octets = host.split(".").map { it.toIntOrNull() }
            val privateIpv4 = octets.size == 4 && octets.all { it != null && it in 0..255 } &&
                (octets[0] == 10 || octets[0] == 127 ||
                    (octets[0] == 192 && octets[1] == 168) ||
                    (octets[0] == 172 && (octets[1] ?: 0) in 16..31))
            require(privateIpv4 || host == "localhost") {
                "HTTPはプライベートIPv4だけに限定しています。外部接続はHTTPSを使用してください"
            }
        }
        return source
    }
}

class IronasuClient(endpoint: String, private val token: String) {
    private val baseUrl = EndpointRules.normalize(endpoint)

    suspend fun get(path: String): JSONObject = execute("GET", path, null)
    suspend fun post(path: String, body: JSONObject): JSONObject = execute("POST", path, body)

    private suspend fun execute(method: String, path: String, body: JSONObject?): JSONObject =
        withContext(Dispatchers.IO) {
            require(path.startsWith("/") && !path.startsWith("//") && !path.contains("..")) {
                "無効なAPIパスです"
            }
            val conn = (URL(baseUrl + path).openConnection() as HttpURLConnection)
            try {
                conn.requestMethod = method
                conn.connectTimeout = 10_000
                conn.readTimeout = 180_000
                conn.setRequestProperty("Accept", "application/json")
                if (token.isNotBlank()) conn.setRequestProperty("Authorization", "Bearer " + token)
                if (body != null) {
                    conn.doOutput = true
                    conn.setRequestProperty("Content-Type", "application/json; charset=utf-8")
                    conn.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }
                }
                val code = conn.responseCode
                val stream = if (code in 200..299) conn.inputStream else conn.errorStream
                val bytes = stream?.use { source ->
                    val buffer = ByteArray(8192)
                    val output = ByteArrayOutputStream()
                    while (true) {
                        val read = source.read(buffer)
                        if (read < 0) break
                        if (output.size() + read > 4 * 1024 * 1024) {
                            throw IllegalStateException("応答が大きすぎます")
                        }
                        output.write(buffer, 0, read)
                    }
                    output.toByteArray()
                } ?: byteArrayOf()
                val response = try { JSONObject(String(bytes, Charsets.UTF_8)) }
                    catch (_: Exception) { throw IllegalStateException("サーバー応答を解析できません (HTTP $code)") }
                if (code !in 200..299) {
                    throw IllegalStateException(response.optJSONObject("error")?.optString("message")
                        ?: "HTTP $code")
                }
                response
            } finally {
                conn.disconnect()
            }
        }

    suspend fun status(): JSONObject = get("/ui/status")
    suspend fun models(provider: String): JSONArray =
        get("/ui/models/installed?provider=" + enc(provider)).optJSONArray("models") ?: JSONArray()
    suspend fun downloads(): JSONArray = get("/ui/models/downloads").optJSONArray("jobs") ?: JSONArray()
    suspend fun download(provider: String, model: String, file: String = ""): JSONObject =
        post("/ui/models/download", JSONObject().put("provider", provider).put("model", model)
            .apply { if (provider == "native") put("file", file) })
    suspend fun loadLmStudio(model: String): JSONObject =
        post("/ui/models/load", JSONObject().put("provider", "lmstudio").put("model", model))
    suspend fun searchGguf(query: String): JSONArray =
        get("/ui/models/native/search?q=" + enc(query)).optJSONArray("models") ?: JSONArray()
    suspend fun ggufFiles(repo: String): JSONArray =
        get("/ui/models/native/files?repo=" + enc(repo)).optJSONArray("files") ?: JSONArray()
    suspend fun startGguf(installedId: String): JSONObject =
        post("/ui/runtime/start", JSONObject().put("id", installedId))

    suspend fun chat(messages: List<ChatLine>): String {
        val entries = JSONArray()
        messages.takeLast(16).forEach { msg ->
            entries.put(JSONObject().put("role", msg.role).put("content", msg.text))
        }
        val request = JSONObject()
            .put("model", "ironasu")
            .put("messages", entries)
            .put("stream", false)
            .put("temperature", 0.2)
            .put("max_tokens", 1024)
            .put("x_ironasu_mode", "auto")
        val response = post("/v1/chat/completions", request)
        val choice = response.optJSONArray("choices")?.optJSONObject(0)?.optJSONObject("message")
            ?: throw IllegalStateException("モデル応答がありません")
        val content = choice.optString("content", "")
        return if (content.isNotBlank()) content else "(空の応答)"
    }

    suspend fun startAgents(corpus: String, objective: String, workers: Int,
                            partitions: Int, modelSlots: Int): JSONObject =
        post("/mobile/agents", JSONObject()
            .put("corpus", corpus).put("objective", objective).put("workers", workers)
            .put("partitions", partitions).put("model_slots", modelSlots)
            .put("model", "auto")).getJSONObject("job")

    suspend fun startHarness(task: String): JSONObject =
        post("/mobile/harness", JSONObject().put("task", task)).getJSONObject("job")
    suspend fun job(id: String): JSONObject = get("/mobile/jobs/" + id).getJSONObject("job")
    suspend fun cancel(id: String): JSONObject =
        post("/mobile/jobs/" + id + "/cancel", JSONObject()).getJSONObject("job")

    private fun enc(value: String): String = URLEncoder.encode(value, "UTF-8")
}

data class ChatLine(val role: String, val text: String)
