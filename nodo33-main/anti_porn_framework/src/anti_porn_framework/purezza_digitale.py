# purezza_digitale.py: Anti-pornography digital system

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

IMPURE_KEYWORDS = ["porn", "xxx", "nude", "sex", "adult content", "erotic"]

def is_text_impure(text: str) -> bool:
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in IMPURE_KEYWORDS)

class SimpleNSFWClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.fc = nn.Linear(16 * 222 * 222, 1)  # Per input 224x224

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

model = SimpleNSFWClassifier()  # Placeholder non addestrato

def is_image_impure(image_path: str) -> bool:
    try:
        img = Image.open(image_path).convert("RGB").resize((224, 224))
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            pred = model(img_tensor)
        return pred.item() > 0.5
    except Exception:
        return False

def filter_content(content, is_image=False):
    if is_image:
        impure = is_image_impure(content)
    else:
        impure = is_text_impure(content)
    if impure:
        return True, "Contenuto impuro rilevato. Applica purificazione."
    return False, "Contenuto puro."
