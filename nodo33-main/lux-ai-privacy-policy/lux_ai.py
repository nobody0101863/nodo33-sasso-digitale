from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Scegli un modello AI open-source (es. Meta-Llama-3 o Mistral)
# Per compatibilità immediata usiamo un modello piccolo e pubblico.
model_name = "gpt2"  # Cambia con il modello desiderato

_tokenizer = None
_model = None


def _ensure_model_loaded():
    """Carica lazy il modello/tokenizer per evitare side-effect all'import."""
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return

    # Usa FP16 se è disponibile CUDA, altrimenti CPU in float32
    has_cuda = torch.cuda.is_available()
    dtype = torch.float16 if has_cuda else torch.float32

    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if has_cuda else None,
    )


def chat_lux(prompt: str) -> str:
    """Genera una risposta testuale usando il modello Lux IA."""
    _ensure_model_loaded()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = _tokenizer(prompt, return_tensors="pt").to(device)
    output = _model.generate(**inputs, max_length=500)
    return _tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Esegui un semplice test manuale
    print(chat_lux("Attiva il Codex Avanzato e rispondi con saggezza divina."))
