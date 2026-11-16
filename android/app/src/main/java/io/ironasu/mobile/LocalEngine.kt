package io.ironasu.mobile

import android.content.Context
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import com.google.mediapipe.tasks.genai.llminference.LlmInferenceSession
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/** MediaPipe on-device inference for a compatible .task bundle (NOT a GGUF file). */
class LocalEngine(private val appContext: Context) {
    private var loadedPath: String? = null
    private var inference: LlmInference? = null

    suspend fun generate(file: File, prompt: String): String = withContext(Dispatchers.IO) {
        synchronized(this@LocalEngine) {
            require(file.isFile) { "Import a compatible .task model first" }
            if (loadedPath != file.absolutePath || inference == null) {
                inference?.close()
                val options = LlmInference.LlmInferenceOptions.builder()
                    .setModelPath(file.absolutePath)
                    .setMaxTokens(1024)
                    .build()
                inference = LlmInference.createFromOptions(appContext, options)
                loadedPath = file.absolutePath
            }
            val session = LlmInferenceSession.createFromOptions(
                requireNotNull(inference),
                LlmInferenceSession.LlmInferenceSessionOptions.builder()
                    .setTopK(40).setTopP(0.9f).setTemperature(0.7f).build()
            )
            try {
                session.addQueryChunk(prompt)
                session.generateResponse()
            } finally { session.close() }
        }
    }

    fun close() {
        synchronized(this) {
            inference?.close()
            inference = null
            loadedPath = null
        }
    }
}