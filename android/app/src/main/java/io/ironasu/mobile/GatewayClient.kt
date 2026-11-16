package io.ironasu.mobile

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URI
import java.net.URLEncoder
import java.net.URL

/** Only explicitly configured ironasu endpoints; does not connect to arbitrary model URLs. */
class GatewayClient(private val baseUrl: String, private val apiKey: String) {
    private val origin: String = validateOrigin(baseUrl)

    suspend fun get(path: String): JSONObject = withContext(Dispatchers.IO) { request("GET", path, null) }
    suspend fun post(path: String, payload: JSONObject): JSONObject = withContext(Dispatchers.IO) {
        request("POST", path, payload)
    }

    private fun request(method: String, path: String, body: JSONObject?): JSONObject {
        require(path.startsWith("/") && !path.startsWith("//")) { "Invalid API path" }
        val conn = URL(origin + path).openConnection() as HttpURLConnection
        try {
            conn.requestMethod = method
            conn.connectTimeout = 10_000
            conn.readTimeout = if (path == "/v1/chat/completions") 650_000 else 90_000
            conn.setRequestProperty("Accept", "application/json")
            conn.setRequestProperty("Cache-Control", "no-store")
            if (apiKey.isNotBlank()) conn.setRequestProperty("Authorization", "Bearer $apiKey")
            if (body != null) {
                conn.doOutput = true
                conn.setRequestProperty("Content-Type", "application/json; charset=utf-8")
                conn.outputStream.use { it.write(body.toString().toByteArray(Charsets.UTF_8)) }
            }
            val stream = if (conn.responseCode in 200..299) conn.inputStream else conn.errorStream
            val bytes = stream?.use { input ->
                val buffer = ByteArray(8192)
                val sink = java.io.ByteArrayOutputStream()
                while (true) {
                    val n = input.read(buffer)
                    if (n < 0) break
                    if (sink.size() + n > 3 * 1024 * 1024) throw IOException("Response too large")
                    sink.write(buffer, 0, n)
                }
                sink.toByteArray()
            } ?: ByteArray(0)
            val output = try { JSONObject(String(bytes, Charsets.UTF_8)) }
                         catch (_: Exception) { JSONObject() }
            if (conn.responseCode !in 200..299) {
                val msg = output.optJSONObject("error")?.optString("message")
                    ?.takeIf { it.isNotBlank() } ?: "HTTP ${conn.responseCode}"
                throw IOException(msg)
            }
            return output
        } finally {
            conn.disconnect()
        }
    }

    suspend fun chat(messages: List<Pair<String, String>>, model: String): String {
        val list = JSONArray()
        for ((role, content) in messages) list.put(JSONObject().put("role", role).put("content", content))
        val data = post("/v1/chat/completions", JSONObject()
            .put("model", model).put("messages", list).put("stream", false)
            .put("max_tokens", 1024).put("x_ironasu_cache", true))
        val message = data.optJSONArray("choices")?.optJSONObject(0)?.optJSONObject("message")
        return message?.optString("content")?.takeIf { it.isNotBlank() }
            ?: throw IOException("Model returned empty content")
    }

    companion object {
        fun encode(value: String): String = URLEncoder.encode(value, "UTF-8")

        fun validateOrigin(url: String): String {
            val uri = try { URI(url.trim()) } catch (e: Exception) { throw IllegalArgumentException("Invalid URL") }
            val scheme = uri.scheme?.lowercase()
            val host = uri.host?.lowercase() ?: throw IllegalArgumentException("Host is required")
            require(scheme == "https" || scheme == "http") { "Only HTTP(S) is supported" }
            require(uri.userInfo == null && uri.query == null && uri.fragment == null &&
                (uri.path.isNullOrEmpty() || uri.path == "/")) { "Enter the server origin only" }
            if (scheme == "http") {
                require(isPrivateHost(host)) {
                    "HTTP is limited to local network addresses. Use HTTPS over the internet."
                }
            }
            return url.trim().trimEnd('/')
        }

        private fun isPrivateHost(host: String): Boolean {
            if (host == "localhost" || host.endsWith(".local")) return true
            val parts = host.split('.')
            if (parts.size != 4 || parts.any { it.toIntOrNull()?.let { n -> n in 0..255 } != true }) return false
            val octets = parts.map { it.toInt() }
            return octets[0] == 10 || (octets[0] == 192 && octets[1] == 168) ||
                (octets[0] == 172 && octets[1] in 16..31) || octets[0] == 127
        }
    }
}