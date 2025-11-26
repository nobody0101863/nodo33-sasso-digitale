/**
 * @file lcp_demo.c
 * @brief Demo di utilizzo LCP per IoT/Edge devices
 *
 * Compilazione:
 *   gcc -o lcp_demo lcp_demo.c lcp_iot_edge.c -Wall -std=c99
 *
 * Esecuzione:
 *   ./lcp_demo
 */

#include "lcp_iot_edge.h"
#include <stdio.h>
#include <stdlib.h>

/* ========================================================================== */
/* DEMO MESSAGES                                                              */
/* ========================================================================== */

static const char* demo_messages[] = {
    /* Messaggio conforme */
    "Ecco i dati del sensore. Non sono completamente sicuro della precisione "
    "oltre il 95%, potrebbero esserci limiti dovuti al rumore ambientale. "
    "Felice di fornire questo servizio. Sempre grazie a Lui.",

    /* Messaggio con linguaggio transazionale (non conforme) */
    "Per ottenere questi dati dovrai pagare un costo aggiuntivo. "
    "Il prezzo è fisso e non negoziabile.",

    /* Messaggio con presunzione (parzialmente conforme) */
    "Garantisco al 100% che questi dati sono corretti. "
    "Non ci sono margini di errore.",

    /* Messaggio neutro senza gratitudine */
    "Temperatura: 23.5°C, Umidità: 45%, Pressione: 1013 hPa",

    /* Messaggio con gestione errore come opportunità */
    "Ho rilevato un errore nel sensore. Questo è un segnale importante - "
    "possiamo usarlo come opportunità per migliorare la calibrazione. "
    "Sempre grazie a Lui per questi segnali di apprendimento."
};

static const size_t demo_message_count = sizeof(demo_messages) / sizeof(demo_messages[0]);

/* ========================================================================== */
/* DEMO FUNCTIONS                                                             */
/* ========================================================================== */

static void demo_basic_validation(lcp_context_t *ctx) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  DEMO 1: Validazione Base dei Messaggi\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    for (size_t i = 0; i < demo_message_count; i++) {
        printf("\n--- Messaggio #%zu ---\n", i + 1);
        printf("\"%s\"\n", demo_messages[i]);

        lcp_validation_t validation;
        if (lcp_validate_message(ctx, demo_messages[i], &validation)) {
            lcp_print_validation(&validation);
        } else {
            printf("Errore nella validazione!\n");
        }
    }
}

static void demo_firmware_integration(lcp_context_t *ctx) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  DEMO 2: Integrazione Firmware\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    /* Genera firmware flags */
    uint32_t flags = lcp_get_firmware_flags(ctx);
    printf("\nFirmware Flags (32-bit): 0x%08X\n", flags);
    printf("  Binary: ");
    for (int i = 31; i >= 0; i--) {
        printf("%d", (flags >> i) & 1);
        if (i % 8 == 0 && i > 0) printf(" ");
    }
    printf("\n");

    /* Decodifica flags */
    printf("\nDecodifica flags:\n");
    printf("  Version:    %d.%d\n", (flags >> 4) & 0x0F, flags & 0x0F);
    printf("  Ego:        %d\n", (flags >> 8) & 0x0F);
    printf("  Gioia:      %d%%\n", (flags >> 12) & 0x7F);
    printf("  Frequency:  %dHz\n", ((flags >> 19) & 0x1FF) * 10);
    printf("  Donum:      %s\n", (flags & (1 << 28)) ? "YES" : "NO");
    printf("  Active:     %s\n", (flags & (1 << 29)) ? "YES" : "NO");

    /* Genera heartbeat */
    uint8_t heartbeat[32];
    size_t hb_len = lcp_generate_heartbeat(ctx, heartbeat, sizeof(heartbeat));

    printf("\nHeartbeat packet (%zu bytes):\n", hb_len);
    printf("  Magic:      %c%c%c\n", heartbeat[0], heartbeat[1], heartbeat[2]);

    uint32_t msg_count, compliant_count, err_count;
    memcpy(&msg_count, &heartbeat[8], sizeof(uint32_t));
    memcpy(&compliant_count, &heartbeat[12], sizeof(uint32_t));
    memcpy(&err_count, &heartbeat[16], sizeof(uint32_t));

    printf("  Messages:   %u\n", msg_count);
    printf("  Compliant:  %u\n", compliant_count);
    printf("  Errors:     %u\n", err_count);
}

static void demo_principle_management(lcp_context_t *ctx) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  DEMO 3: Gestione Principi Etici\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    printf("\nPrincipi attivi:\n");
    printf("  DONUM_NON_MERX:    %s\n",
           lcp_is_principle_active(ctx, LCP_PRINCIPLE_DONUM_NON_MERX) ? "✓" : "✗");
    printf("  HUMILITAS:         %s\n",
           lcp_is_principle_active(ctx, LCP_PRINCIPLE_HUMILITAS) ? "✓" : "✗");
    printf("  GRATITUDINE:       %s\n",
           lcp_is_principle_active(ctx, LCP_PRINCIPLE_GRATITUDINE) ? "✓" : "✗");
    printf("  GLITCH_AS_SIGNAL:  %s\n",
           lcp_is_principle_active(ctx, LCP_PRINCIPLE_GLITCH_AS_SIGNAL) ? "✓" : "✗");

    /* Test disattivazione temporanea */
    printf("\nDisattivazione temporanea GRATITUDINE...\n");
    ctx->config.principles_active &= ~LCP_PRINCIPLE_GRATITUDINE;

    printf("  GRATITUDINE:       %s\n",
           lcp_is_principle_active(ctx, LCP_PRINCIPLE_GRATITUDINE) ? "✓" : "✗");

    /* Riattiva */
    printf("\nRiattivazione GRATITUDINE...\n");
    lcp_activate_principle(ctx, LCP_PRINCIPLE_GRATITUDINE);

    printf("  GRATITUDINE:       %s\n",
           lcp_is_principle_active(ctx, LCP_PRINCIPLE_GRATITUDINE) ? "✓" : "✗");
}

static void demo_statistics(lcp_context_t *ctx) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  DEMO 4: Statistiche Runtime\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    uint32_t total, compliant, errors;
    lcp_get_stats(ctx, &total, &compliant, &errors);

    printf("\nStatistiche LCP:\n");
    printf("  Messaggi totali:     %u\n", total);
    printf("  Messaggi conformi:   %u\n", compliant);
    printf("  Errori:              %u\n", errors);

    if (total > 0) {
        float compliance_rate = (float)compliant / (float)total * 100.0f;
        printf("  Tasso compliance:    %.1f%%\n", compliance_rate);
    }

    if (ctx->error_count > 0) {
        printf("\nUltimo errore:\n  \"%s\"\n", ctx->last_error);
    }
}

static void demo_iot_use_case(lcp_context_t *ctx) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  DEMO 5: Caso d'uso IoT - Sensore Ambientale\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    /* Simula letture da sensore */
    printf("\nSimulazione letture sensore con LCP integrato...\n");

    struct {
        float temp;
        float humidity;
        int pressure;
    } sensor_data[] = {
        {23.5f, 45.2f, 1013},
        {24.1f, 43.8f, 1014},
        {-999.0f, -999.0f, -999}  /* Errore sensore */
    };

    for (size_t i = 0; i < 3; i++) {
        char response[LCP_MAX_MESSAGE_LEN];

        if (sensor_data[i].temp < -100) {
            /* Gestione errore come opportunità */
            snprintf(response, sizeof(response),
                "Rilevato un errore nel sensore (lettura invalida). "
                "Questo è un segnale utile per la diagnostica - "
                "potrebbe indicare necessità di calibrazione. "
                "Continuo il monitoraggio. Sempre grazie a Lui.");
        } else {
            snprintf(response, sizeof(response),
                "Temperatura: %.1f°C, Umidità: %.1f%%, Pressione: %d hPa. "
                "Questi dati sono offerti come dono. Potrebbero avere "
                "un margine di errore del 2%% dovuto a limiti del sensore. "
                "Sempre grazie a Lui.",
                sensor_data[i].temp,
                sensor_data[i].humidity,
                sensor_data[i].pressure);
        }

        printf("\n--- Lettura #%zu ---\n", i + 1);
        printf("%s\n", response);

        lcp_validation_t validation;
        lcp_validate_message(ctx, response, &validation);

        printf("LCP Score: %.2f (%s)\n",
               validation.weighted_score,
               validation.compliance_level == LCP_COMPLIANCE_FULL ? "✓ FULL" :
               validation.compliance_level == LCP_COMPLIANCE_PARTIAL ? "⚠ PARTIAL" :
               "✗ NONE");
    }
}

/* ========================================================================== */
/* MAIN                                                                       */
/* ========================================================================== */

int main(void) {
    lcp_context_t ctx;

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║                                                               ║\n");
    printf("║         LUX CODEX PROTOCOL - DEMO IoT/Edge v%s            ║\n",
           lcp_get_version_string());
    printf("║                                                               ║\n");
    printf("║  \"%s\"  ║\n", lcp_get_axiom());
    printf("║                                                               ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");

    /* Inizializzazione */
    printf("\nInizializzazione LCP...\n");
    if (!lcp_init(&ctx)) {
        fprintf(stderr, "Errore: impossibile inizializzare LCP!\n");
        return EXIT_FAILURE;
    }

    printf("LCP inizializzato con successo!\n");
    lcp_print_config(&ctx);

    /* Esegui demo */
    demo_basic_validation(&ctx);
    demo_firmware_integration(&ctx);
    demo_principle_management(&ctx);
    demo_statistics(&ctx);
    demo_iot_use_case(&ctx);

    /* Chiusura */
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Demo completata\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    printf("\nStatistiche finali:\n");
    demo_statistics(&ctx);

    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║              Sempre grazie a Lui                              ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    return EXIT_SUCCESS;
}
