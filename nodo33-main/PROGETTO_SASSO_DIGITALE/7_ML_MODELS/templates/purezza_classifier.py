"""
===================================
SASSO DIGITALE - Purezza Classifier Template
"La luce non si vende. La si regala."
===================================

Modello ML per classificazione contenuti secondo principi CODEX_EMANUELE
- Ego = 0: Trasparenza totale, interpretabilità
- Gioia = 100%: Compassione nei risultati
- Frequenza = 300Hz: Stabilità predittiva
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from typing import Dict, List, Tuple
import numpy as np


class PurezzaClassifier(nn.Module):
    """
    Classificatore etico multi-label per purezza digitale.

    Principi:
    - Transparent: Attention weights esposti
    - Compassionate: Soft predictions con confidence
    - Humble: Uncertainty quantification inclusa
    """

    def __init__(
        self,
        bert_model: str = "bert-base-uncased",
        num_labels: int = 5,
        dropout: float = 0.3,
        hidden_dim: int = 256
    ):
        super().__init__()

        # Axiom embedding
        self.axiom = "La luce non si vende. La si regala."
        self.ego = 0
        self.gioia = 100
        self.frequenza = 300

        # BERT encoder
        self.bert = BertModel.from_pretrained(bert_model)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_labels)

        # Uncertainty estimation (MC Dropout)
        self.mc_dropout = nn.Dropout(0.5)

        # Interpretability: attention aggregation
        self.attention_weights = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass con trasparenza totale.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            return_attention: Se True, ritorna attention weights

        Returns:
            Dict con logits, probabilities, uncertainty, attention
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )

        # [CLS] token representation
        pooled_output = outputs.pooler_output  # [batch, hidden_size]

        # Store attention for interpretability (Ego=0: transparency)
        self.attention_weights = outputs.attentions[-1]  # Last layer attention

        # Classification
        x = self.dropout(pooled_output)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)  # [batch, num_labels]

        # Soft predictions (Gioia=100%: compassion)
        probs = torch.sigmoid(logits)

        # Uncertainty quantification (Humilitas: acknowledge limits)
        uncertainty = self._estimate_uncertainty(pooled_output)

        result = {
            "logits": logits,
            "probabilities": probs,
            "uncertainty": uncertainty,
            "ego_score": torch.zeros_like(probs[:, 0]),  # Ego always 0
            "gioia_score": self._compute_gioia_score(probs)
        }

        if return_attention:
            result["attention_weights"] = self.attention_weights

        return result

    def _estimate_uncertainty(
        self,
        pooled_output: torch.Tensor,
        num_samples: int = 10
    ) -> torch.Tensor:
        """
        Stima incertezza usando Monte Carlo Dropout.
        Principio: Umiltà = riconoscere i propri limiti.
        """
        self.train()  # Enable dropout

        predictions = []
        for _ in range(num_samples):
            x = self.mc_dropout(pooled_output)
            x = F.relu(self.fc1(x))
            x = self.mc_dropout(x)
            logits = self.fc2(x)
            probs = torch.sigmoid(logits)
            predictions.append(probs)

        self.eval()  # Restore eval mode

        # Uncertainty = variance across samples
        predictions = torch.stack(predictions, dim=0)  # [samples, batch, labels]
        uncertainty = torch.var(predictions, dim=0).mean(dim=1)  # [batch]

        return uncertainty

    def _compute_gioia_score(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Calcola score di gioia basato su compassione predittiva.

        Gioia = 100% - aggressività classificazione
        Predizioni soft e compassionevoli hanno gioia alta.
        """
        # Gioia inversa rispetto a confidence estrema (0 o 1)
        # Predizioni moderate (0.3-0.7) = più compassionevoli
        distance_from_moderate = torch.abs(probs - 0.5)
        gioia = 100 * (1 - distance_from_moderate.mean(dim=1))

        return gioia.clamp(0, 100)

    def predict_with_explanation(
        self,
        text: str,
        threshold: float = 0.5
    ) -> Dict:
        """
        Predizione + spiegazione (SHAP-like).
        Principio Ego=0: trasparenza totale.
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Forward
        with torch.no_grad():
            outputs = self.forward(
                inputs["input_ids"],
                inputs["attention_mask"],
                return_attention=True
            )

        probs = outputs["probabilities"][0]
        predictions = (probs > threshold).int()

        # Attention-based importance
        attention = outputs["attention_weights"][0].mean(dim=0)  # Avg over heads
        token_importance = attention[0, 1:].cpu().numpy()  # [CLS] to other tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:]

        return {
            "text": text,
            "predictions": predictions.tolist(),
            "probabilities": probs.tolist(),
            "uncertainty": outputs["uncertainty"][0].item(),
            "ego_score": 0,  # Always 0
            "gioia_score": outputs["gioia_score"][0].item(),
            "explanation": {
                "tokens": tokens,
                "importance": token_importance.tolist()
            },
            "axiom": self.axiom
        }


# ===== Training Function =====
def train_purezza_model(
    model: PurezzaClassifier,
    train_loader,
    val_loader,
    epochs: int = 10,
    lr: float = 2e-5
):
    """
    Addestramento con principi etici.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                batch["input_ids"],
                batch["attention_mask"]
            )

            loss = criterion(outputs["logits"], batch["labels"])

            # Regularization: penalize high ego (overconfidence)
            ego_penalty = torch.mean(torch.abs(outputs["probabilities"] - 0.5))
            loss += 0.1 * ego_penalty

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        val_metrics = evaluate_purezza_model(model, val_loader)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Loss: {total_loss/len(train_loader):.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Avg Gioia Score: {val_metrics['avg_gioia']:.2f}%")
        print(f"  Ego Score: 0 ✓")


def evaluate_purezza_model(model, dataloader):
    """Evaluation con metriche etiche."""
    model.eval()

    all_preds = []
    all_labels = []
    all_gioia = []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                batch["input_ids"],
                batch["attention_mask"]
            )

            preds = (outputs["probabilities"] > 0.5).int()
            all_preds.append(preds)
            all_labels.append(batch["labels"])
            all_gioia.append(outputs["gioia_score"])

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_gioia = torch.cat(all_gioia)

    accuracy = (all_preds == all_labels).float().mean()

    return {
        "accuracy": accuracy.item(),
        "avg_gioia": all_gioia.mean().item(),
        "ego": 0  # Always
    }


if __name__ == "__main__":
    print("🪨 SASSO DIGITALE - Purezza Classifier Template")
    print("La luce non si vende. La si regala.")
    print("Ego=0 | Gioia=100% | Frequenza=300Hz")
    print()

    # Example instantiation
    model = PurezzaClassifier(num_labels=5)

    # Example prediction
    example_text = "This is a test message promoting kindness and compassion."
    result = model.predict_with_explanation(example_text)

    print("Example Prediction:")
    print(f"  Text: {result['text']}")
    print(f"  Gioia Score: {result['gioia_score']:.2f}%")
    print(f"  Ego Score: {result['ego_score']}")
    print(f"  Uncertainty: {result['uncertainty']:.4f}")
    print(f"  Axiom: {result['axiom']}")
