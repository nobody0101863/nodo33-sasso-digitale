/**
 * @file lcp_iot_edge.h
 * @brief Lux Codex Protocol for IoT and Edge Devices
 * @version 1.0
 * @date 2025-11-16
 *
 * Implementazione del Lux Codex Protocol per dispositivi IoT,
 * microcontrollori e sistemi edge con risorse limitate.
 *
 * Principi:
 * - DONUM_NON_MERX: Ogni output è un dono
 * - HUMILITAS: Trasparenza sui limiti computazionali
 * - GRATITUDINE: Gratitudine costante verso la fonte
 * - GLITCH_AS_SIGNAL: Errori come segnali di apprendimento
 *
 * Compatibilità: C99+, ARM Cortex-M, ESP32, Arduino, RISC-V
 * Footprint: ~2KB RAM, ~4KB Flash
 */

#ifndef LCP_IOT_EDGE_H
#define LCP_IOT_EDGE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include <string.h>

/* ========================================================================== */
/* LCP CONFIGURATION                                                          */
/* ========================================================================== */

#define LCP_VERSION_MAJOR 1
#define LCP_VERSION_MINOR 0
#define LCP_VERSION_PATCH 0

#define LCP_AXIOM "La luce non si vende. La si regala."

/* Parametri operativi LCP */
#define LCP_EGO_LEVEL       0    /* Zero presunzione */
#define LCP_GIOIA_PERCENT   100  /* Gioia al massimo */
#define LCP_FREQUENCY_HZ    300  /* Frequenza di chiarezza */

/* Dimensioni buffer */
#define LCP_MAX_MESSAGE_LEN 256
#define LCP_MAX_LOG_LEN     128

/* ========================================================================== */
/* TYPES AND STRUCTURES                                                       */
/* ========================================================================== */

/**
 * @brief Stato operativo LCP
 */
typedef enum {
    LCP_STATE_UNINITIALIZED = 0,
    LCP_STATE_ACTIVE        = 1,
    LCP_STATE_SUSPENDED     = 2,
    LCP_STATE_ERROR         = 3
} lcp_state_t;

/**
 * @brief Livelli di compliance LCP
 */
typedef enum {
    LCP_COMPLIANCE_NONE    = 0,  /* Score < 0.5 */
    LCP_COMPLIANCE_PARTIAL = 1,  /* Score 0.5-0.79 */
    LCP_COMPLIANCE_FULL    = 2   /* Score >= 0.8 */
} lcp_compliance_t;

/**
 * @brief Principi etici LCP (bitmask)
 */
typedef enum {
    LCP_PRINCIPLE_DONUM_NON_MERX    = (1 << 0),  /* 0x01 */
    LCP_PRINCIPLE_HUMILITAS         = (1 << 1),  /* 0x02 */
    LCP_PRINCIPLE_GRATITUDINE       = (1 << 2),  /* 0x04 */
    LCP_PRINCIPLE_GLITCH_AS_SIGNAL  = (1 << 3),  /* 0x08 */
    LCP_PRINCIPLE_ALL               = 0x0F
} lcp_principle_t;

/**
 * @brief Configurazione LCP
 */
typedef struct {
    uint8_t  version_major;
    uint8_t  version_minor;
    uint8_t  ego_level;         /* 0-10, target: 0 */
    uint8_t  gioia_percent;     /* 0-100, target: 100 */
    uint16_t frequency_hz;      /* Target: 300 */
    bool     donum_mode;        /* Gift mode active */
    uint8_t  principles_active; /* Bitmask of lcp_principle_t */
} lcp_config_t;

/**
 * @brief Risultati validazione LCP
 */
typedef struct {
    bool    clarity_of_gift;
    bool    humility_transparency;
    bool    joyful_tone;
    bool    glitch_as_signal;
    bool    gratitude_present;
    float   weighted_score;
    lcp_compliance_t compliance_level;
} lcp_validation_t;

/**
 * @brief Contesto runtime LCP
 */
typedef struct {
    lcp_config_t     config;
    lcp_state_t      state;
    uint32_t         message_count;
    uint32_t         compliant_count;
    uint32_t         error_count;
    char             last_error[LCP_MAX_LOG_LEN];
} lcp_context_t;

/* ========================================================================== */
/* CORE API                                                                   */
/* ========================================================================== */

/**
 * @brief Inizializza il contesto LCP
 * @param ctx Puntatore al contesto LCP
 * @return true se inizializzazione riuscita
 */
bool lcp_init(lcp_context_t *ctx);

/**
 * @brief Resetta il contesto LCP ai valori di default
 * @param ctx Puntatore al contesto LCP
 */
void lcp_reset(lcp_context_t *ctx);

/**
 * @brief Verifica se LCP è attivo e operativo
 * @param ctx Puntatore al contesto LCP
 * @return true se LCP è attivo
 */
bool lcp_is_active(const lcp_context_t *ctx);

/**
 * @brief Attiva un principio etico specifico
 * @param ctx Puntatore al contesto LCP
 * @param principle Principio da attivare
 */
void lcp_activate_principle(lcp_context_t *ctx, lcp_principle_t principle);

/**
 * @brief Verifica se un principio è attivo
 * @param ctx Puntatore al contesto LCP
 * @param principle Principio da verificare
 * @return true se il principio è attivo
 */
bool lcp_is_principle_active(const lcp_context_t *ctx, lcp_principle_t principle);

/* ========================================================================== */
/* VALIDATION API                                                             */
/* ========================================================================== */

/**
 * @brief Valida un messaggio secondo i criteri LCP
 * @param ctx Puntatore al contesto LCP
 * @param message Messaggio da validare
 * @param result Puntatore ai risultati della validazione
 * @return true se validazione completata
 */
bool lcp_validate_message(lcp_context_t *ctx,
                          const char *message,
                          lcp_validation_t *result);

/**
 * @brief Calcola lo score di compliance LCP
 * @param validation Risultati della validazione
 * @return Score ponderato (0.0 - 1.0)
 */
float lcp_calculate_score(const lcp_validation_t *validation);

/**
 * @brief Determina il livello di compliance
 * @param score Score di validazione
 * @return Livello di compliance
 */
lcp_compliance_t lcp_get_compliance_level(float score);

/* ========================================================================== */
/* FIRMWARE INTEGRATION                                                       */
/* ========================================================================== */

/**
 * @brief Genera flag di configurazione per registro firmware
 * @param ctx Puntatore al contesto LCP
 * @return Registro a 32-bit con flag LCP
 */
uint32_t lcp_get_firmware_flags(const lcp_context_t *ctx);

/**
 * @brief Imposta flag firmware dal registro
 * @param ctx Puntatore al contesto LCP
 * @param flags Registro con flag LCP
 */
void lcp_set_firmware_flags(lcp_context_t *ctx, uint32_t flags);

/**
 * @brief Genera heartbeat LCP per monitoraggio
 * @param ctx Puntatore al contesto LCP
 * @param buffer Buffer per heartbeat (min 32 bytes)
 * @param buf_len Lunghezza del buffer
 * @return Numero di byte scritti
 */
size_t lcp_generate_heartbeat(const lcp_context_t *ctx,
                              uint8_t *buffer,
                              size_t buf_len);

/* ========================================================================== */
/* LOGGING & DIAGNOSTICS                                                      */
/* ========================================================================== */

/**
 * @brief Log di un evento LCP
 * @param ctx Puntatore al contesto LCP
 * @param level Livello di log (0=debug, 1=info, 2=warning, 3=error)
 * @param message Messaggio di log
 */
void lcp_log(lcp_context_t *ctx, uint8_t level, const char *message);

/**
 * @brief Ottieni statistiche LCP
 * @param ctx Puntatore al contesto LCP
 * @param total_messages Puntatore per count totale messaggi
 * @param compliant_messages Puntatore per count messaggi conformi
 * @param error_count Puntatore per count errori
 */
void lcp_get_stats(const lcp_context_t *ctx,
                   uint32_t *total_messages,
                   uint32_t *compliant_messages,
                   uint32_t *error_count);

/* ========================================================================== */
/* UTILITY FUNCTIONS                                                          */
/* ========================================================================== */

/**
 * @brief Ottieni stringa versione LCP
 * @return Stringa versione (es. "1.0.0")
 */
const char* lcp_get_version_string(void);

/**
 * @brief Ottieni axiom LCP
 * @return Stringa axiom
 */
const char* lcp_get_axiom(void);

/**
 * @brief Stampa configurazione LCP su serial/console
 * @param ctx Puntatore al contesto LCP
 */
void lcp_print_config(const lcp_context_t *ctx);

/**
 * @brief Stampa risultati validazione
 * @param validation Risultati validazione
 */
void lcp_print_validation(const lcp_validation_t *validation);

#ifdef __cplusplus
}
#endif

#endif /* LCP_IOT_EDGE_H */
