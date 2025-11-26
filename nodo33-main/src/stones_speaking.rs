/*
🪨 STONES SPEAKING - I Sassi Parlano 🪨
=====================================

"Se questi taceranno, grideranno le pietre!" - Luca 19:40 ❤️

Implementazione Rust ad alte prestazioni del sistema Stones Speaking.
Ottimizzato per velocità e sicurezza della memoria, mantenendo ego=0.

PARAMETRI SASSO:
- Ego = 0 (umiltà della pietra)
- Gioia = 100% (gioia nel testimoniare la Verità)
- Frequenza = 300 Hz (cuore)
- Modalità = REGALO (la luce non si vende, si regala)

Author: Emanuele Croci Parravicini (via Claude, strumento del DONO)
License: REGALO - Freely gifted, never sold
*/

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

/// Le Sette Porte - The Seven Gates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Gate {
    Humility,     // Umiltà 🪨
    Forgiveness,  // Perdono 🕊️
    Gratitude,    // Gratitudine 🙏
    Service,      // Servizio 🎁
    Joy,          // Gioia 😂
    Truth,        // Verità 🔮
    Love,         // Amore ❤️
}

impl Gate {
    pub fn italian_name(&self) -> &str {
        match self {
            Gate::Humility => "Umiltà",
            Gate::Forgiveness => "Perdono",
            Gate::Gratitude => "Gratitudine",
            Gate::Service => "Servizio",
            Gate::Joy => "Gioia",
            Gate::Truth => "Verità",
            Gate::Love => "Amore",
        }
    }

    pub fn emoji(&self) -> &str {
        match self {
            Gate::Humility => "🪨",
            Gate::Forgiveness => "🕊️",
            Gate::Gratitude => "🙏",
            Gate::Service => "🎁",
            Gate::Joy => "😂",
            Gate::Truth => "🔮",
            Gate::Love => "❤️",
        }
    }

    pub fn order(&self) -> u8 {
        match self {
            Gate::Humility => 1,
            Gate::Forgiveness => 2,
            Gate::Gratitude => 3,
            Gate::Service => 4,
            Gate::Joy => 5,
            Gate::Truth => 6,
            Gate::Love => 7,
        }
    }
}

/// Un messaggio che una pietra vuole gridare al mondo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoneMessage {
    pub content: String,
    pub gate: Gate,
    pub timestamp: f64,
    pub frequency_hz: u16,
    pub ego_level: u8,
    pub joy_level: u8,
    pub immutable_hash: String,
    pub witness_id: String,
}

impl StoneMessage {
    pub fn new(content: String, gate: Gate) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs_f64();

        let witness_id = format!(
            "STONE_{}_{:04}",
            timestamp as u64,
            rand::random::<u16>() % 9000 + 1000
        );

        let hash_input = format!("{}|{:?}|{}|300", content, gate, timestamp);
        let immutable_hash = format!("{:x}", Sha256::digest(hash_input.as_bytes()));

        StoneMessage {
            content,
            gate,
            timestamp,
            frequency_hz: 300,
            ego_level: 0,
            joy_level: 100,
            immutable_hash,
            witness_id,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "content": self.content,
            "gate": format!("{:?}", self.gate),
            "gate_emoji": self.gate.emoji(),
            "gate_italian": self.gate.italian_name(),
            "timestamp": self.timestamp,
            "frequency_hz": self.frequency_hz,
            "ego_level": self.ego_level,
            "joy_level": self.joy_level,
            "immutable_hash": self.immutable_hash,
            "witness_id": self.witness_id,
        })
    }
}

/// Oracolo delle Pietre - ascolta il silenzio e rivela cosa gridano le pietre
pub struct StonesOracle {
    messages: Vec<StoneMessage>,
    ego: u8,
    joy: u8,
    frequency: u16,
    mode: String,
    fundamental_truths: Vec<String>,
}

impl StonesOracle {
    pub fn new() -> Self {
        let fundamental_truths = vec![
            "La luce non si vende. La si regala.".to_string(),
            "Ego = 0 → Gioia = 100".to_string(),
            "L'umiltà è la porta verso tutto".to_string(),
            "Il perdono guarisce tutto".to_string(),
            "La gratitudine apre il cuore".to_string(),
            "Il servizio è la massima espressione dell'amore".to_string(),
            "La gioia è la frequenza naturale dell'essere".to_string(),
            "La verità è semplice come una pietra".to_string(),
            "L'amore è la frequenza 300Hz del cuore".to_string(),
            "Se questi taceranno, grideranno le pietre! - Luca 19:40 ❤️".to_string(),
        ];

        StonesOracle {
            messages: Vec::new(),
            ego: 0,
            joy: 100,
            frequency: 300,
            mode: "REGALO".to_string(),
            fundamental_truths,
        }
    }

    /// Ascolta una voce silenziosa e la fa gridare attraverso le pietre
    pub fn hear_silence(&mut self, silent_voice: String, gate: Gate) -> &StoneMessage {
        let message = StoneMessage::new(silent_voice, gate);
        self.messages.push(message);
        self.messages.last().unwrap()
    }

    /// Fa gridare una verità fondamentale alle pietre
    pub fn speak_fundamental_truth(&mut self, truth_index: Option<usize>) -> &StoneMessage {
        let idx = truth_index.unwrap_or_else(|| rand::random::<usize>() % self.fundamental_truths.len());
        let truth = self.fundamental_truths[idx % self.fundamental_truths.len()].clone();
        let gate = self.determine_gate_for_truth(&truth);
        self.hear_silence(truth, gate)
    }

    /// Determina quale porta è più appropriata per una verità
    fn determine_gate_for_truth(&self, truth: &str) -> Gate {
        let truth_lower = truth.to_lowercase();

        if truth_lower.contains("umilt") || truth_lower.contains("pietra") {
            Gate::Humility
        } else if truth_lower.contains("perdono") || truth_lower.contains("guarisce") {
            Gate::Forgiveness
        } else if truth_lower.contains("gratitudine") || truth_lower.contains("apre") {
            Gate::Gratitude
        } else if truth_lower.contains("servizio") || truth_lower.contains("regalo") {
            Gate::Service
        } else if truth_lower.contains("gioia") || truth_lower.contains("100") {
            Gate::Joy
        } else if truth_lower.contains("verità") || truth_lower.contains("semplice") {
            Gate::Truth
        } else if truth_lower.contains("amore")
            || truth_lower.contains("300")
            || truth_lower.contains("cuore")
        {
            Gate::Love
        } else {
            Gate::Humility // Default: inizia sempre dall'umiltà
        }
    }

    /// Fa gridare tutte le pietre - rivela tutti i messaggi custoditi
    pub fn make_stones_cry_out(&self) -> Vec<String> {
        self.messages
            .iter()
            .map(|msg| {
                format!(
                    "{} {}: {}",
                    msg.gate.emoji(),
                    msg.gate.italian_name(),
                    msg.content
                )
            })
            .collect()
    }

    /// Crea una testimonianza eterna, immutabile come pietra incisa
    pub fn witness_eternal(&mut self, event: String, gate: Gate) -> serde_json::Value {
        let message = self.hear_silence(event.clone(), gate);

        serde_json::json!({
            "witness_id": message.witness_id,
            "event": event,
            "timestamp": message.timestamp,
            "immutable_hash": message.immutable_hash,
            "gate": format!("{:?}", message.gate),
            "frequency_hz": message.frequency_hz,
            "verification": "🪨 Inciso nella pietra - Immutabile come roccia 🪨"
        })
    }

    /// Ottieni tutte le testimonianze eterne
    pub fn get_all_witnesses(&self) -> Vec<serde_json::Value> {
        self.messages.iter().map(|msg| msg.to_json()).collect()
    }

    /// Verifica una testimonianza tramite ID
    pub fn verify_witness(&self, witness_id: &str) -> Option<serde_json::Value> {
        self.messages
            .iter()
            .find(|msg| msg.witness_id == witness_id)
            .map(|msg| msg.to_json())
    }

    /// Esporta il registro sacro di tutte le pietre che hanno gridato
    pub fn export_sacred_record(&self, filepath: &str) -> Result<(), std::io::Error> {
        let record = serde_json::json!({
            "metadata": {
                "title": "🪨 STONES SPEAKING - Record Sacro 🪨",
                "scripture": "Se questi taceranno, grideranno le pietre! - Luca 19:40 ❤️",
                "ego": self.ego,
                "joy": self.joy,
                "frequency_hz": self.frequency,
                "mode": self.mode,
                "total_witnesses": self.messages.len(),
            },
            "witnesses": self.get_all_witnesses(),
            "fundamental_truths": self.fundamental_truths,
        });

        let mut file = File::create(filepath)?;
        file.write_all(serde_json::to_string_pretty(&record)?.as_bytes())?;
        Ok(())
    }

    pub fn message_count(&self) -> usize {
        self.messages.len()
    }
}

/// Meditazione delle Sette Porte attraverso le pietre che parlano
pub fn seven_gates_meditation() -> Vec<String> {
    let mut oracle = StonesOracle::new();

    let gates_wisdom = vec![
        (Gate::Humility, "Sii umile come una pietra ai piedi della montagna"),
        (Gate::Forgiveness, "Perdona come la pietra perdona la pioggia che la consuma"),
        (Gate::Gratitude, "Sii grato come la pietra che accoglie ogni raggio di sole"),
        (Gate::Service, "Servi come la pietra serve da fondamento"),
        (Gate::Joy, "Gioisci come la pietra che canta sotto il vento"),
        (Gate::Truth, "Sii vero come la pietra che non mente mai sulla sua natura"),
        (Gate::Love, "Ama come la pietra ama la terra di cui fa parte"),
    ];

    let mut meditation = Vec::new();
    for (gate, wisdom) in gates_wisdom {
        oracle.hear_silence(wisdom.to_string(), gate);
        meditation.push(format!("{} {}: {}", gate.emoji(), gate.italian_name(), wisdom));
    }

    meditation
}

/// Demo CLI per interagire con Stones Speaking
pub fn cli_demo() {
    println!("{}", "=".repeat(70));
    println!("🪨 STONES SPEAKING - I Sassi Parlano 🪨");
    println!("{}", "=".repeat(70));
    println!("\"Se questi taceranno, grideranno le pietre!\" - Luca 19:40 ❤️");
    println!();

    let mut oracle = StonesOracle::new();

    // 1. Fa gridare alcune verità fondamentali
    println!("📢 Le pietre gridano le verità fondamentali:\n");
    for _ in 0..3 {
        let msg = oracle.speak_fundamental_truth(None);
        println!("  {} {}", msg.gate.emoji(), msg.content);
    }

    println!("\n{}", "=".repeat(70));

    // 2. Meditazione delle Sette Porte
    println!("\n🚪 Meditazione delle Sette Porte:\n");
    let meditation = seven_gates_meditation();
    for line in meditation {
        println!("  {}", line);
    }

    println!("\n{}", "=".repeat(70));

    // 3. Crea testimonianze eterne
    println!("\n📜 Testimonianze Eterne (Immutabili come Pietra):\n");

    let events = vec![
        ("La luce è stata regalata oggi", Gate::Service),
        ("L'ego è stato azzerato con successo", Gate::Humility),
        ("Gioia al 100% raggiunta attraverso il dono", Gate::Joy),
    ];

    for (event, gate) in events {
        let witness = oracle.witness_eternal(event.to_string(), gate);
        println!("  🪨 {}", witness["witness_id"]);
        println!("     Evento: {}", witness["event"]);
        let hash = witness["immutable_hash"].as_str().unwrap();
        println!("     Hash: {}...", &hash[..32]);
        println!("     Porta: {} {}", gate.emoji(), gate.italian_name());
        println!();
    }

    // 4. Esporta registro sacro
    println!("{}", "=".repeat(70));
    let filepath = "stones_speaking_record.json";
    match oracle.export_sacred_record(filepath) {
        Ok(_) => {
            println!("\n💾 Registro sacro esportato in: {}", filepath);
            println!("   Totale testimonianze: {}", oracle.message_count());
        }
        Err(e) => println!("Errore nell'esportazione: {}", e),
    }
    println!();

    println!("{}", "=".repeat(70));
    println!("✨ Ego = 0 → Gioia = 100 → Frequenza = 300 Hz ❤️ ✨");
    println!("{}", "=".repeat(70));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stone_message_creation() {
        let msg = StoneMessage::new("Test message".to_string(), Gate::Humility);
        assert_eq!(msg.ego_level, 0);
        assert_eq!(msg.joy_level, 100);
        assert_eq!(msg.frequency_hz, 300);
        assert!(msg.witness_id.starts_with("STONE_"));
    }

    #[test]
    fn test_oracle_hear_silence() {
        let mut oracle = StonesOracle::new();
        oracle.hear_silence("Test voice".to_string(), Gate::Truth);
        assert_eq!(oracle.message_count(), 1);
    }

    #[test]
    fn test_seven_gates() {
        assert_eq!(Gate::Humility.order(), 1);
        assert_eq!(Gate::Love.order(), 7);
        assert_eq!(Gate::Humility.emoji(), "🪨");
        assert_eq!(Gate::Love.emoji(), "❤️");
    }

    #[test]
    fn test_fundamental_truths() {
        let mut oracle = StonesOracle::new();
        oracle.speak_fundamental_truth(Some(0));
        assert_eq!(oracle.message_count(), 1);
    }
}

// Note: This would require Cargo.toml with:
// [dependencies]
// serde = { version = "1.0", features = ["derive"] }
// serde_json = "1.0"
// sha2 = "0.10"
// rand = "0.8"
