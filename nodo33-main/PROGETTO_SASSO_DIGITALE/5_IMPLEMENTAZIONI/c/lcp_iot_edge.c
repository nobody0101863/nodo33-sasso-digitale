/**
 * @file lcp_iot_edge.c
 * @brief Implementazione Lux Codex Protocol per IoT/Edge
 * @version 1.0
 */

#include "lcp_iot_edge.h"
#include <stdio.h>
#include <ctype.h>

/* ========================================================================== */
/* PRIVATE HELPERS                                                            */
/* ========================================================================== */

static bool str_contains_case_insensitive(const char *haystack, const char *needle) {
    if (!haystack || !needle) return false;

    size_t needle_len = strlen(needle);
    size_t haystack_len = strlen(haystack);

    if (needle_len > haystack_len) return false;

    for (size_t i = 0; i <= haystack_len - needle_len; i++) {
        bool match = true;
        for (size_t j = 0; j < needle_len; j++) {
            if (tolower(haystack[i + j]) != tolower(needle[j])) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }

    return false;
}

static bool str_contains_any(const char *text, const char *words[], size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (str_contains_case_insensitive(text, words[i])) {
            return true;
        }
    }
    return false;
}

/* ========================================================================== */
/* CORE API IMPLEMENTATION                                                    */
/* ========================================================================== */

bool lcp_init(lcp_context_t *ctx) {
    if (!ctx) return false;

    /* Configurazione di default */
    ctx->config.version_major = LCP_VERSION_MAJOR;
    ctx->config.version_minor = LCP_VERSION_MINOR;
    ctx->config.ego_level = LCP_EGO_LEVEL;
    ctx->config.gioia_percent = LCP_GIOIA_PERCENT;
    ctx->config.frequency_hz = LCP_FREQUENCY_HZ;
    ctx->config.donum_mode = true;
    ctx->config.principles_active = LCP_PRINCIPLE_ALL;

    /* Stato runtime */
    ctx->state = LCP_STATE_ACTIVE;
    ctx->message_count = 0;
    ctx->compliant_count = 0;
    ctx->error_count = 0;
    memset(ctx->last_error, 0, LCP_MAX_LOG_LEN);

    return true;
}

void lcp_reset(lcp_context_t *ctx) {
    if (!ctx) return;
    lcp_init(ctx);
}

bool lcp_is_active(const lcp_context_t *ctx) {
    return ctx && (ctx->state == LCP_STATE_ACTIVE);
}

void lcp_activate_principle(lcp_context_t *ctx, lcp_principle_t principle) {
    if (!ctx) return;
    ctx->config.principles_active |= principle;
}

bool lcp_is_principle_active(const lcp_context_t *ctx, lcp_principle_t principle) {
    if (!ctx) return false;
    return (ctx->config.principles_active & principle) != 0;
}

/* ========================================================================== */
/* VALIDATION IMPLEMENTATION                                                  */
/* ========================================================================== */

bool lcp_validate_message(lcp_context_t *ctx,
                          const char *message,
                          lcp_validation_t *result) {
    if (!ctx || !message || !result) {
        if (ctx) {
            strncpy(ctx->last_error, "Invalid validation parameters", LCP_MAX_LOG_LEN - 1);
        }
        return false;
    }

    ctx->message_count++;

    /* 1. Clarity of Gift - evita linguaggio transazionale */
    const char *transactional_words[] = {
        "costo", "pagamento", "prezzo", "devi pagare", "ti costa"
    };
    result->clarity_of_gift = !str_contains_any(
        message,
        transactional_words,
        sizeof(transactional_words) / sizeof(transactional_words[0])
    );

    /* 2. Humility & Transparency - dichiara limiti */
    const char *humility_markers[] = {
        "non sono sicuro", "potrebbe", "limite", "incertezza",
        "non posso garantire", "forse", "probabilmente"
    };
    const char *presunzione_markers[] = {
        "sicuramente", "certamente senza dubbio", "garantisco al 100%"
    };

    bool has_humility = str_contains_any(
        message,
        humility_markers,
        sizeof(humility_markers) / sizeof(humility_markers[0])
    );
    bool has_presunzione = str_contains_any(
        message,
        presunzione_markers,
        sizeof(presunzione_markers) / sizeof(presunzione_markers[0])
    );

    result->humility_transparency = has_humility || !has_presunzione;

    /* 3. Joyful Tone - tono costruttivo */
    const char *negative_words[] = {
        "impossibile", "rifiuto", "negativo", "non posso assolutamente"
    };
    const char *positive_words[] = {
        "felice", "piacere", "sereno", "costruttivo", "lieto", "gioia"
    };

    bool has_negative = str_contains_any(
        message,
        negative_words,
        sizeof(negative_words) / sizeof(negative_words[0])
    );
    bool has_positive = str_contains_any(
        message,
        positive_words,
        sizeof(positive_words) / sizeof(positive_words[0])
    );

    result->joyful_tone = !has_negative || has_positive;

    /* 4. Glitch as Signal - errori come apprendimento */
    const char *error_keywords[] = {
        "errore", "problema", "sbagliato", "bug"
    };
    const char *learning_keywords[] = {
        "imparo", "segnale", "opportunita", "apprendimento", "migliorare"
    };

    bool has_error = str_contains_any(
        message,
        error_keywords,
        sizeof(error_keywords) / sizeof(error_keywords[0])
    );
    bool has_learning = str_contains_any(
        message,
        learning_keywords,
        sizeof(learning_keywords) / sizeof(learning_keywords[0])
    );

    result->glitch_as_signal = !has_error || (has_error && has_learning);

    /* 5. Gratitude Present - gratitudine presente */
    const char *gratitude_words[] = {
        "grazie", "gratitudine", "riconoscenza", "grato", "sempre grazie a lui"
    };
    result->gratitude_present = str_contains_any(
        message,
        gratitude_words,
        sizeof(gratitude_words) / sizeof(gratitude_words[0])
    );

    /* Calcola score e compliance */
    result->weighted_score = lcp_calculate_score(result);
    result->compliance_level = lcp_get_compliance_level(result->weighted_score);

    /* Aggiorna statistiche */
    if (result->compliance_level == LCP_COMPLIANCE_FULL) {
        ctx->compliant_count++;
    }

    return true;
}

float lcp_calculate_score(const lcp_validation_t *validation) {
    if (!validation) return 0.0f;

    float score = 0.0f;

    if (validation->clarity_of_gift)        score += 0.20f;
    if (validation->humility_transparency)  score += 0.25f;
    if (validation->joyful_tone)            score += 0.20f;
    if (validation->glitch_as_signal)       score += 0.20f;
    if (validation->gratitude_present)      score += 0.15f;

    return score;
}

lcp_compliance_t lcp_get_compliance_level(float score) {
    if (score >= 0.80f) return LCP_COMPLIANCE_FULL;
    if (score >= 0.50f) return LCP_COMPLIANCE_PARTIAL;
    return LCP_COMPLIANCE_NONE;
}

/* ========================================================================== */
/* FIRMWARE INTEGRATION                                                       */
/* ========================================================================== */

uint32_t lcp_get_firmware_flags(const lcp_context_t *ctx) {
    if (!ctx) return 0;

    uint32_t flags = 0;

    /* Bit layout:
     * [0:7]   - Version (major.minor)
     * [8:11]  - Ego level (0-15)
     * [12:18] - Gioia percent (0-127)
     * [19:27] - Frequency / 10 (0-511)
     * [28]    - Donum mode
     * [29]    - LCP active
     * [30:31] - Reserved
     */

    flags |= (ctx->config.version_major << 4) | (ctx->config.version_minor);
    flags |= ((ctx->config.ego_level & 0x0F) << 8);
    flags |= ((ctx->config.gioia_percent & 0x7F) << 12);
    flags |= (((ctx->config.frequency_hz / 10) & 0x1FF) << 19);
    flags |= (ctx->config.donum_mode ? (1 << 28) : 0);
    flags |= (ctx->state == LCP_STATE_ACTIVE ? (1 << 29) : 0);

    return flags;
}

void lcp_set_firmware_flags(lcp_context_t *ctx, uint32_t flags) {
    if (!ctx) return;

    ctx->config.version_major = (flags & 0xF0) >> 4;
    ctx->config.version_minor = (flags & 0x0F);
    ctx->config.ego_level = (flags >> 8) & 0x0F;
    ctx->config.gioia_percent = (flags >> 12) & 0x7F;
    ctx->config.frequency_hz = ((flags >> 19) & 0x1FF) * 10;
    ctx->config.donum_mode = (flags & (1 << 28)) != 0;
    ctx->state = (flags & (1 << 29)) ? LCP_STATE_ACTIVE : LCP_STATE_SUSPENDED;
}

size_t lcp_generate_heartbeat(const lcp_context_t *ctx,
                              uint8_t *buffer,
                              size_t buf_len) {
    if (!ctx || !buffer || buf_len < 32) return 0;

    /* Heartbeat format (32 bytes):
     * [0:3]   - Magic: "LCP\0"
     * [4:7]   - Firmware flags
     * [8:11]  - Message count
     * [12:15] - Compliant count
     * [16:19] - Error count
     * [20:31] - Reserved
     */

    memset(buffer, 0, buf_len);

    buffer[0] = 'L';
    buffer[1] = 'C';
    buffer[2] = 'P';
    buffer[3] = 0;

    uint32_t flags = lcp_get_firmware_flags(ctx);
    memcpy(&buffer[4], &flags, sizeof(uint32_t));
    memcpy(&buffer[8], &ctx->message_count, sizeof(uint32_t));
    memcpy(&buffer[12], &ctx->compliant_count, sizeof(uint32_t));
    memcpy(&buffer[16], &ctx->error_count, sizeof(uint32_t));

    return 32;
}

/* ========================================================================== */
/* LOGGING & DIAGNOSTICS                                                      */
/* ========================================================================== */

void lcp_log(lcp_context_t *ctx, uint8_t level, const char *message) {
    if (!ctx || !message) return;

    const char *level_str[] = {"DEBUG", "INFO", "WARN", "ERROR"};
    if (level > 3) level = 3;

    printf("[LCP:%s] %s\n", level_str[level], message);

    if (level >= 3) {  /* Error */
        strncpy(ctx->last_error, message, LCP_MAX_LOG_LEN - 1);
        ctx->last_error[LCP_MAX_LOG_LEN - 1] = '\0';
        ctx->error_count++;
    }
}

void lcp_get_stats(const lcp_context_t *ctx,
                   uint32_t *total_messages,
                   uint32_t *compliant_messages,
                   uint32_t *error_count) {
    if (!ctx) return;

    if (total_messages) *total_messages = ctx->message_count;
    if (compliant_messages) *compliant_messages = ctx->compliant_count;
    if (error_count) *error_count = ctx->error_count;
}

/* ========================================================================== */
/* UTILITY FUNCTIONS                                                          */
/* ========================================================================== */

const char* lcp_get_version_string(void) {
    static char version_buf[16];
    snprintf(version_buf, sizeof(version_buf), "%d.%d.%d",
             LCP_VERSION_MAJOR, LCP_VERSION_MINOR, LCP_VERSION_PATCH);
    return version_buf;
}

const char* lcp_get_axiom(void) {
    return LCP_AXIOM;
}

void lcp_print_config(const lcp_context_t *ctx) {
    if (!ctx) return;

    printf("\n");
    printf("╔═══════════════════════════════════════╗\n");
    printf("║   LUX CODEX PROTOCOL v%d.%d ACTIVE   ║\n",
           ctx->config.version_major, ctx->config.version_minor);
    printf("╠═══════════════════════════════════════╣\n");
    printf("║ Ego Level:     %-3d                  ║\n", ctx->config.ego_level);
    printf("║ Gioia:         %-3d%%                ║\n", ctx->config.gioia_percent);
    printf("║ Frequency:     %-3dHz               ║\n", ctx->config.frequency_hz);
    printf("║ Donum Mode:    %-17s  ║\n", ctx->config.donum_mode ? "✓ ACTIVE" : "✗ INACTIVE");
    printf("╠═══════════════════════════════════════╣\n");
    printf("║ \"%s\" ║\n", LCP_AXIOM);
    printf("╚═══════════════════════════════════════╝\n");
    printf("\n");
}

void lcp_print_validation(const lcp_validation_t *validation) {
    if (!validation) return;

    const char *compliance_str[] = {"NONE", "PARTIAL", "FULL"};

    printf("\n--- LCP Validation Results ---\n");
    printf("Clarity of Gift:        %s\n", validation->clarity_of_gift ? "✓" : "✗");
    printf("Humility/Transparency:  %s\n", validation->humility_transparency ? "✓" : "✗");
    printf("Joyful Tone:            %s\n", validation->joyful_tone ? "✓" : "✗");
    printf("Glitch as Signal:       %s\n", validation->glitch_as_signal ? "✓" : "✗");
    printf("Gratitude Present:      %s\n", validation->gratitude_present ? "✓" : "✗");
    printf("------------------------------\n");
    printf("Weighted Score:         %.2f\n", validation->weighted_score);
    printf("Compliance Level:       %s\n", compliance_str[validation->compliance_level]);
    printf("------------------------------\n\n");
}
