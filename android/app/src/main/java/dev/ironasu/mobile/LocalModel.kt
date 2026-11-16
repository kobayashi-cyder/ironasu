package dev.ironasu.mobile

import android.content.Context
import android.net.Uri
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/** Real on-device inference for MediaPipe-compatible .task models.
 * GGUF files downloaded on the PC are not compatible with this runner.
 */
class LocalModel(private val context: Context) {
    private val mutex = Mutex()
    private var engine: LlmInference? = null
    private val modelFile: File get() = File(context.filesDir, "models/active.task")

    fun hasModel(): Boolean = modelFile.isFile

    suspend fun importAndLoad(uri: Uri) = mutex.withLock {
        withContext(Dispatchers.IO) {
            val folder = modelFile.parentFile!!
            check(folder.exists() || folder.mkdirs()) { "保存先を作成できません" }
            val temp = File(folder, "incoming.task")
            try {
                context.contentResolver.openInputStream(uri)?.use { input ->
                    temp.outputStream().use { output -> input.copyTo(output, 512 * 1024) }
                } ?: error("モデルファイルを開けません")
                check(temp.length() > 0L) { "モデルファイルが空です" }
                engine?.close()
                engine = null
                if (modelFile.exists()) check(modelFile.delete()) { "旧モデルを削除できません" }
                check(temp.renameTo(modelFile)) { "モデルを保存できません" }
                createEngine()
            } finally {
                temp.delete()
            }
        }
    }

    suspend fun loadExisting() = mutex.withLock {
        withContext(Dispatchers.IO) {
            check(modelFile.isFile) { "端末内モデルを選択してください" }
            if (engine == null) createEngine()
        }
    }

    private fun createEngine() {
        val options = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelFile.absolutePath)
            .setMaxTokens(2048)
            .setMaxTopK(64)
            .build()
        engine = LlmInference.createFromOptions(context.applicationContext, options)
    }

    suspend fun respond(prompt: String): String = mutex.withLock {
        withContext(Dispatchers.IO) {
            check(modelFile.isFile) { "端末内モデルを先にインポートしてください" }
            if (engine == null) createEngine()
            engine!!.generateResponse(prompt)
        }
    }

    suspend fun close() = mutex.withLock {
        withContext(Dispatchers.IO) { engine?.close(); engine = null }
    }
}
