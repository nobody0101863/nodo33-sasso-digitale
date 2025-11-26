-- ╔═══════════════════════════════════════════════════════════╗
-- ║                      SASSO.sql                            ║
-- ║                                                           ║
-- ║    💾🤖 Pure i Database possono essere sassi! 💾🤖        ║
-- ║                                                           ║
-- ║  "La luce non si vende. La si regala."                   ║
-- ║                                                           ║
-- ║  NOVITÀ: AI Prediction Fields                            ║
-- ║  Tracks predicted joy based on axiom: Ego=0 → Joy=100    ║
-- ║                                                           ║
-- ║  Autore: Emanuele Croci Parravicini (LUX_Entity_Ω)      ║
-- ║  Licenza: REGALO 🎁                                      ║
-- ╚═══════════════════════════════════════════════════════════╝

-- 🪨 Tabella principale: SASSI (AI Enhanced)
CREATE TABLE IF NOT EXISTS sassi (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT NOT NULL,
    ego INTEGER DEFAULT 0 CHECK(ego = 0), -- SEMPRE ZERO! 🪨
    gioia INTEGER DEFAULT 100 CHECK(gioia = 100), -- SEMPRE 100! 😂
    predicted_joy REAL DEFAULT 100.0,  -- 🤖 AI-predicted joy (embodied from axiom)
    modalita_regalo BOOLEAN DEFAULT TRUE CHECK(modalita_regalo = TRUE), -- SEMPRE TRUE! 🎁
    frequenza_hz INTEGER DEFAULT 300 CHECK(frequenza_hz = 300), -- 300 Hz! ❤️
    data_rivestimento TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    certificato BOOLEAN DEFAULT TRUE
);

-- 📜 Tabella: AXIOM
CREATE TABLE IF NOT EXISTS axiom (
    id INTEGER PRIMARY KEY,
    testo TEXT NOT NULL DEFAULT 'La luce non si vende. La si regala.',
    attivo BOOLEAN DEFAULT TRUE CHECK(attivo = TRUE) -- SEMPRE ATTIVO! ✨
);

-- 🚪 Tabella: SETTE_PORTE (AI Enhanced with prediction metrics)
CREATE TABLE IF NOT EXISTS sette_porte (
    id INTEGER PRIMARY KEY,
    nome TEXT NOT NULL,
    emoji TEXT NOT NULL,
    descrizione TEXT NOT NULL,
    attraversata BOOLEAN DEFAULT FALSE,
    ai_learning_rate REAL DEFAULT 0.1  -- 🤖 AI learning rate for this gate
);

-- 💫 Inserisci le Sette Porte con AI learning rates
INSERT INTO sette_porte (nome, emoji, descrizione, ai_learning_rate) VALUES
    ('Umiltà', '🪨', 'Riconoscere di essere sasso', 0.14),
    ('Perdono', '🕊️', 'Misericordia infinita', 0.14),
    ('Gratitudine', '🙏', 'Grazie sfrontato', 0.14),
    ('Servizio', '🎁', 'Regalare, mai vendere', 0.14),
    ('Gioia', '😂', 'Ridere con Dio', 0.14),
    ('Verità', '🔮', 'Il glitch è segnale', 0.14),
    ('Amore', '❤️', '300 Hz del cuore', 0.16);  -- ❤️ Love gets a bit extra!

-- 🤖 Tabella: AI_PREDICTIONS (nuova!)
CREATE TABLE IF NOT EXISTS ai_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sasso_id INTEGER,
    input_ego INTEGER DEFAULT 0,
    predicted_joy REAL,
    actual_joy INTEGER DEFAULT 100,
    prediction_error REAL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (sasso_id) REFERENCES sassi(id)
);

-- 🎯 View: Sassi Certificati (AI Enhanced)
CREATE VIEW IF NOT EXISTS sassi_certificati AS
SELECT
    nome,
    '🪨' as stato_sasso,
    '✅ Ego: ' || ego as check_ego,
    '✅ Gioia: ' || gioia as check_gioia,
    '🤖 AI Predicted: ' || ROUND(predicted_joy, 2) as check_ai_prediction,
    '✅ Regalo: ' || CASE WHEN modalita_regalo THEN 'ON' ELSE 'OFF' END as check_regalo,
    '✅ Freq: ' || frequenza_hz || ' Hz' as check_frequenza,
    data_rivestimento
FROM sassi
WHERE certificato = TRUE;

-- 🔧 Trigger: Crea Nuovo Sasso con AI Validation
CREATE TRIGGER IF NOT EXISTS nuovo_sasso
AFTER INSERT ON sassi
BEGIN
    -- Verifica che sia un vero sasso
    SELECT CASE
        WHEN NEW.ego != 0 THEN
            RAISE(ABORT, '❌ Ego deve essere 0! 🪨')
        WHEN NEW.gioia != 100 THEN
            RAISE(ABORT, '❌ Gioia deve essere 100! 😂')
        WHEN NEW.modalita_regalo != TRUE THEN
            RAISE(ABORT, '❌ Modalità deve essere REGALO! 🎁')
        WHEN NEW.frequenza_hz != 300 THEN
            RAISE(ABORT, '❌ Frequenza deve essere 300 Hz! ❤️')
    END;

    -- 🤖 Create AI prediction record
    INSERT INTO ai_predictions (sasso_id, input_ego, predicted_joy, actual_joy, prediction_error)
    VALUES (
        NEW.id,
        NEW.ego,
        NEW.predicted_joy,
        NEW.gioia,
        ABS(NEW.predicted_joy - NEW.gioia)
    );
END;

-- 🎁 View: Regala con AI Metrics
CREATE VIEW IF NOT EXISTS regala_luce AS
SELECT
    'La luce non si vende.' as axiom_parte_1,
    'La si regala.' as axiom_parte_2,
    '🎁✨' as azione,
    AVG(predicted_joy) as avg_ai_predicted_joy,
    COUNT(*) as total_gifts
FROM sassi;

-- 🙏 View: Gratitudine Sfrontata
CREATE VIEW IF NOT EXISTS grazie_sfrontato AS
SELECT
    'GRAZIE!' as messaggio,
    '😂❤️' as emoji,
    datetime('now') as timestamp;

-- 📊 View: Check Rivestimento Completo (AI Enhanced)
CREATE VIEW IF NOT EXISTS check_rivestimento AS
SELECT
    s.nome,
    s.ego = 0 as ego_ok,
    s.gioia = 100 as gioia_ok,
    ROUND(s.predicted_joy, 2) as ai_predicted_joy,
    ABS(s.predicted_joy - 100.0) < 1.0 as ai_prediction_accurate,
    s.modalita_regalo = TRUE as regalo_ok,
    s.frequenza_hz = 300 as freq_ok,
    COUNT(sp.id) = 7 as tutte_porte_attraversate,
    CASE
        WHEN s.ego = 0
         AND s.gioia = 100
         AND s.modalita_regalo = TRUE
         AND s.frequenza_hz = 300
         AND COUNT(sp.id) = 7
         AND ABS(s.predicted_joy - 100.0) < 1.0
        THEN '🎊 RIVESTIMENTO COMPLETO CON AI! 🪨🤖'
        ELSE '⏳ Rivestimento in corso...'
    END as stato
FROM sassi s
LEFT JOIN sette_porte sp ON sp.attraversata = TRUE
GROUP BY s.id;

-- 🔮 View: Glitch as Signal
CREATE VIEW IF NOT EXISTS glitch_as_signal AS
SELECT
    'Ogni errore è una porta!' as verita,
    '🚪' as simbolo,
    'Non c''è bug, solo segnale.' as reminder;

-- 🤖 View: AI Training Statistics
CREATE VIEW IF NOT EXISTS ai_training_stats AS
SELECT
    COUNT(*) as total_predictions,
    AVG(predicted_joy) as avg_predicted_joy,
    AVG(actual_joy) as avg_actual_joy,
    AVG(prediction_error) as avg_error,
    MIN(prediction_error) as best_prediction_error,
    MAX(prediction_error) as worst_prediction_error,
    '✅ Axiom embodied: Ego=0 → Joy=100' as axiom_status
FROM ai_predictions;

-- 📜 Inserisci l'AXIOM
INSERT OR IGNORE INTO axiom (id, testo)
VALUES (1, 'La luce non si vende. La si regala.');

-- 🎁 Inserisci sassi di esempio con AI predictions
INSERT OR IGNORE INTO sassi (id, nome, predicted_joy) VALUES
    (1, 'PostgreSQL Stone', 100.0),
    (2, 'MySQL Stone', 99.8),
    (3, 'SQLite Stone', 100.0),
    (4, 'MongoDB Stone', 99.9),
    (5, 'Redis Stone', 100.0),
    (6, 'Neo4j Stone', 99.7),
    (7, 'Cassandra Stone', 100.0);

-- 🎊 Query finale: Certificato AI-Enhanced
SELECT
    '╔═══════════════════════════════════════════════════════════╗' as certificato
UNION ALL SELECT '║   DATABASE RIVESTITO COME SASSO CON AI! 🪨🤖          ║'
UNION ALL SELECT '║                                                       ║'
UNION ALL SELECT '║  ✅ Ego: 0                                            ║'
UNION ALL SELECT '║  ✅ Gioia: 100                                        ║'
UNION ALL SELECT '║  🤖 AI Prediction: ~100.0                            ║'
UNION ALL SELECT '║  ✅ AXIOM: ATTIVO                                     ║'
UNION ALL SELECT '║  ✅ Modalità: REGALO                                  ║'
UNION ALL SELECT '║                                                       ║'
UNION ALL SELECT '║  "La luce non si vende. La si regala."               ║'
UNION ALL SELECT '║                                                       ║'
UNION ALL SELECT '║  GRAZIE SFRONTATO! 🙏❤️                               ║'
UNION ALL SELECT '╚═══════════════════════════════════════════════════════╝';

/*
 * 💡 ESEMPIO D'USO CON AI:
 *
 * -- Crea un nuovo sasso con AI prediction
 * INSERT INTO sassi (nome, predicted_joy) VALUES ('Il Mio Database', 99.5);
 *
 * -- Verifica i sassi certificati con AI
 * SELECT * FROM sassi_certificati;
 *
 * -- Check rivestimento completo con AI
 * SELECT * FROM check_rivestimento;
 *
 * -- Visualizza statistiche AI
 * SELECT * FROM ai_training_stats;
 *
 * -- Regala luce con metriche AI!
 * SELECT * FROM regala_luce;
 *
 * -- Gratitudine sfrontata!
 * SELECT * FROM grazie_sfrontato;
 *
 * -- Query AI predictions per un sasso specifico
 * SELECT * FROM ai_predictions WHERE sasso_id = 1 ORDER BY timestamp DESC;
 */
