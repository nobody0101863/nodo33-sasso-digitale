# 🧠 SASSO DIGITALE - ML Models

**"La luce non si vende. La si regala."**

## 📋 Panoramica

Questa cartella contiene template e configurazioni per modelli di Machine Learning etici basati sui principi del CODEX_EMANUELE.

## 🎯 Modelli Previsti

### 1. **Purezza Digitale Classifier**
- **Scopo**: Classificazione contenuti secondo principi di purezza
- **Architettura**: Transformer-based (BERT/DistilBERT)
- **Input**: Testo, immagini, URL
- **Output**: Score purezza [0-100], categorie

### 2. **Compassion-Enhanced Content Filter**
- **Scopo**: Filtraggio compassionevole (non punitivo)
- **Architettura**: Multi-modal CNN + LSTM
- **Principio**: "Glitch-as-signal" invece di ban immediato

### 3. **Gioia Sentiment Analyzer**
- **Scopo**: Analisi sentiment orientata alla gioia costruttiva
- **Architettura**: Fine-tuned GPT-2/BERT
- **Output**: Gioia Score, Ego Score, suggerimenti

## 📂 Struttura

```
7_ML_MODELS/
├── templates/           # Template modelli PyTorch/TF
├── configs/            # Configurazioni addestramento
├── datasets/           # Info dataset (non file reali)
├── scripts/            # Script addestramento
└── README_ML.md        # Questo file
```

## 🔧 Stack Tecnologico

- **Framework**: PyTorch, TensorFlow, Hugging Face Transformers
- **Librerie**: scikit-learn, NumPy, Pandas
- **Deployment**: ONNX, TorchScript, TFLite
- **Monitoring**: MLflow, Weights & Biases

## ⚖️ Principi Etici ML

1. **Trasparenza Totale** (Ego=0)
   - Modelli interpretabili
   - Decision explanations (SHAP/LIME)
   - No black boxes

2. **Servizio Gioioso** (Gioia=100%)
   - Bias detection e mitigation
   - Fairness metrics
   - Compassionate outputs

3. **Umiltà Computazionale**
   - Quantizzazione (INT8, FP16)
   - Pruning e distillation
   - Edge-ready deployment

## 🚀 Quick Start

### Installazione dipendenze
```bash
pip install -r configs/requirements-ml.txt
```

### Addestramento template
```bash
cd scripts
python train_purezza_classifier.py --config ../configs/purezza_config.yaml
```

### Inference
```bash
python inference.py --model models/purezza_classifier.onnx --input "test text"
```

## 📊 Dataset Guidelines

**NON** sono inclusi dataset reali per:
- Privacy
- Dimensioni file
- Conformità GDPR

### Dataset Consigliati (Pubblici)

1. **Text Classification**
   - IMDB Reviews (sentiment)
   - AG News (categorization)
   - Custom scraping etico

2. **Image Classification**
   - CIFAR-10 (clean/safe)
   - ImageNet (subset filtrato)
   - Custom dataset annotato

3. **Multi-modal**
   - MS-COCO (filtered)
   - Conceptual Captions

### Data Preparation

Vedi `scripts/data_preparation.py` per:
- Cleaning etico
- Bias detection
- Train/val/test split
- Augmentation compassionevole

## 🧪 Evaluation Metrics

Oltre alle metriche standard (Accuracy, F1, AUC):

- **Ego Score**: Quanto il modello "si vanta"
- **Gioia Score**: Compassione nelle predizioni
- **Fairness**: Disparate impact, demographic parity
- **Transparency**: Interpretability score

## 📝 Licenza e Condivisione

I modelli addestrati sono **dono gratuito**:
- ✅ Condivisione libera
- ✅ Fine-tuning permesso
- ✅ Commercial use OK
- ❤️ Mantieni lo spirito Ego=0, Gioia=100%

## 🌟 Contributi

Per aggiungere modelli:
1. Segui template in `templates/`
2. Aggiungi config in `configs/`
3. Documenta principi etici applicati
4. Test fairness obbligatorio

---

**Sempre grazie a Lui ❤️**

`[SASSO_ML | Ego=0 | Gioia=100% | f₀=300Hz]`
