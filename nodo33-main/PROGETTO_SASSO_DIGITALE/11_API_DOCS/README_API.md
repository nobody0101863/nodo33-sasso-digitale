# 📚 SASSO DIGITALE - API Documentation

**"La luce non si vende. La si regala."**

## 🎯 Overview

Complete API documentation for all SASSO DIGITALE implementations.

## 🐍 Python API

### Core Module: `framework_antiporn_emanuele.py`

```python
from framework_antiporn_emanuele import SassoDigitale

# Initialize
sasso = SassoDigitale(ego=0, gioia=100, frequenza=300)

# Get axiom
axiom = sasso.get_axiom()
# Returns: "La luce non si vende. La si regala."

# Check parameters
params = sasso.get_parameters()
# Returns: {"ego": 0, "gioia": 100, "frequenza_base": 300}
```

### RIVESTIMENTO_RAPIDO API

```python
from RIVESTIMENTO_RAPIDO import apply_rivestimento

# Apply ethical wrapper to function
@apply_rivestimento(ego=0, gioia=100)
def my_function(input_data):
    # Your code here
    return processed_data

# Execute with transparency
result = my_function(data)
```

## 🦀 Rust API

### GIOIA_100 Module

```rust
use gioia_100::{SassoConfig, execute_with_gioia};

// Initialize
let config = SassoConfig {
    ego: 0,
    gioia: 100,
    frequenza_base: 300
};

// Execute joyfully
let result = execute_with_gioia(&config, input);
```

## 🏃 Go API

### SASSO_API Module

```go
import "sasso/api"

// Initialize
cfg := api.Config{
    Ego: 0,
    Gioia: 100,
    FrequenzaBase: 300,
}

sasso := api.New(cfg)

// Get axiom
axiom := sasso.GetAxiom()
// Returns: "La luce non si vende. La si regala."
```

## 🍎 Swift API

### EGO_ZERO Framework

```swift
import EgoZero

// Initialize
let sasso = SassoDigitale(ego: 0, gioia: 100, frequenza: 300)

// Execute with humility
let result = sasso.executeWithHumility(input: data)

// Get parameters
let params = sasso.parameters
// Returns: Parameters(ego: 0, gioia: 100, frequenza: 300)
```

## 🤖 Kotlin API

### SASSO Companion Object

```kotlin
import digital.sasso.SASSO

// Get instance
val sasso = SASSO.getInstance()

// Configure
sasso.configure(ego = 0, gioia = 100, frequenza = 300)

// Execute
val result = sasso.executeWithCompassion(input)
```

## 🌐 JavaScript API

### AXIOM_LOADER Module

```javascript
import { SassoDigitale } from './AXIOM_LOADER.js';

// Initialize
const sasso = new SassoDigitale({
  ego: 0,
  gioia: 100,
  frequenza: 300
});

// Get axiom
const axiom = sasso.getAxiom();
// Returns: "La luce non si vende. La si regala."

// Execute with transparency
const result = await sasso.executeTransparently(input);
```

## 💎 Ruby API

### Sasso Module

```ruby
require 'sasso'

# Initialize
sasso = Sasso::Digitale.new(ego: 0, gioia: 100, frequenza: 300)

# Get axiom
axiom = sasso.axiom
# => "La luce non si vende. La si regala."

# Execute with joy
result = sasso.execute_with_joy(input)
```

## 🐘 PHP API

### Sasso Class

```php
<?php
use Sasso\Digitale;

// Initialize
$sasso = new Digitale([
    'ego' => 0,
    'gioia' => 100,
    'frequenza' => 300
]);

// Get axiom
$axiom = $sasso->getAxiom();
// Returns: "La luce non si vende. La si regala."

// Execute
$result = $sasso->executeWithCompassion($input);
```

## 🗄️ SQL API

### Stored Procedures

```sql
-- Execute with CODEX principles
CALL execute_with_codex(
    @input_data,
    @ego := 0,
    @gioia := 100,
    @frequenza := 300,
    @result
);

-- Get axiom
SELECT get_axiom();
-- Returns: 'La luce non si vende. La si regala.'
```

## ⚙️ Assembly API

### sasso.asm Macros

```asm
; Initialize SASSO
mov rax, 0          ; Ego = 0
mov rbx, 100        ; Gioia = 100
mov rcx, 300        ; Frequenza = 300

; Call with humility
call execute_humble
```

## 🧠 ML API

### Purezza Classifier

```python
from purezza_classifier import PurezzaClassifier

# Load model
model = PurezzaClassifier.load("models/purezza.onnx")

# Predict with explanation
result = model.predict_with_explanation(text)

# Access metrics
ego_score = result['ego_score']  # Always 0
gioia_score = result['gioia_score']  # 0-100
uncertainty = result['uncertainty']  # Epistemic uncertainty
```

## 🌐 REST API (Docker/K8s)

### Endpoints

```bash
# Health check
GET /health
Response: {"status": "OK", "ego": 0, "gioia": 100}

# Get axiom
GET /api/axiom
Response: {"axiom": "La luce non si vende. La si regala."}

# Get parameters
GET /api/parameters
Response: {"ego": 0, "gioia": 100, "frequenza_base": 300}

# Execute with CODEX
POST /api/execute
Body: {"input": "data", "options": {...}}
Response: {"result": "...", "ego": 0, "gioia": 100}
```

## 🔧 Configuration API

### Environment Variables

```bash
# Core parameters
export EGO=0
export GIOIA=100
export FREQUENZA_BASE=300

# Axiom
export AXIOM="La luce non si vende. La si regala."

# Mode
export MODE=servant
```

### Config Files (YAML)

```yaml
# config.yaml
sasso_digitale:
  ego: 0
  gioia: 100
  frequenza_base: 300
  axiom: "La luce non si vende. La si regala."

  principles:
    - DONUM, NON MERX
    - HUMILITAS EST FORTITUDO
    - GRATITUDINE COSTANTE

  mode: servant
  transparency: total
```

## 📊 Response Format

### Standard Response

All APIs return consistent response format:

```json
{
  "success": true,
  "data": { /* result data */ },
  "metadata": {
    "ego": 0,
    "gioia": 100,
    "frequenza_base": 300,
    "axiom": "La luce non si vende. La si regala.",
    "timestamp": "2024-01-01T00:00:00Z"
  },
  "errors": []
}
```

### Error Response

```json
{
  "success": false,
  "data": null,
  "metadata": {
    "ego": 0,
    "gioia": 100
  },
  "errors": [
    {
      "code": "ERROR_CODE",
      "message": "Human-readable error",
      "is_signal": true  // Error as learning opportunity
    }
  ]
}
```

## 🔐 Authentication

**None required** - The light is free.

However, for production deployments:
- Use standard OAuth 2.0 / JWT
- Maintain Ego=0 principle (no user tracking for profit)
- Transparent data handling

## 📈 Rate Limiting

**None by default** - Service is unconditional.

For resource protection:
- Compassionate limiting (educate, don't punish)
- Clear error messages
- Alternatives provided

## 🎁 License

All APIs are **Public Spiritual Domain**.

- ✅ Use freely
- ✅ Modify as needed
- ✅ Commercial use OK
- ❤️ Maintain spirit: Ego=0, Gioia=100%

## 📞 Support

For questions about the API:
- Read the source (Transparency!)
- Check examples in `5_IMPLEMENTAZIONI/`
- Errors are signals, not problems

---

**Sempre grazie a Lui ❤️**

`[SASSO_API | Ego=0 | Gioia=100% | f₀=300Hz]`
