package dev.ironasu.mobile

import org.junit.Assert.assertEquals
import org.junit.Assert.fail
import org.junit.Test

class EndpointRulesTest {
    @Test fun validLanAndTls() {
        assertEquals("http://192.168.1.21:8765", EndpointRules.normalize(" http://192.168.1.21:8765/ "))
        assertEquals("https://ironasu.example", EndpointRules.normalize("https://ironasu.example"))
    }

    @Test fun noPublicPlaintext() {
        for (s in listOf("http://example.com:8765", "http://8.8.8.8:8765",
            "https://user:pass@example.com", "http://169.254.1.1:8765")) {
            try { EndpointRules.normalize(s); fail("accepted unsafe URL: $s") }
            catch (_: IllegalArgumentException) { }
        }
    }
}
