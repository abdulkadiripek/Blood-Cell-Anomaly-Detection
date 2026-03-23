"""
Blood Cell Anomaly Detection - PyTorch Model Definition
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path


class BloodClassification(nn.Module):
    """
    Neural network model for blood cell disease classification.
    Architecture: Linear(21,25) -> GELU -> Linear(25,25) -> GELU -> Linear(25,8)
    """
    def __init__(self):
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(21, 25),
            nn.GELU(),
            nn.Linear(25, 25),
            nn.GELU(),
            nn.Linear(25, 8)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


# Disease category labels (LabelEncoder order - alphabetical)
DISEASE_CATEGORIES = [
    'Anemia',
    'Artefact',
    'Infection',
    'Leukemia',
    'Normal_Platelet',
    'Normal_RBC',
    'Normal_WBC',
    'Sickle_Cell_Anemia'
]

# Turkish translations for display
DISEASE_CATEGORIES_TR = {
    'Anemia': 'Anemi (Kansızlık)',
    'Artefact': 'Artefakt (Yapay Bulgu)',
    'Infection': 'Enfeksiyon',
    'Leukemia': 'Lösemi',
    'Normal_Platelet': 'Normal Trombosit',
    'Normal_RBC': 'Normal Kırmızı Kan Hücresi',
    'Normal_WBC': 'Normal Beyaz Kan Hücresi',
    'Sickle_Cell_Anemia': 'Orak Hücre Anemisi'
}

# Severity/risk labels for display
DISEASE_SEVERITY = {
    'Anemia': 'warning',
    'Artefact': 'info',
    'Infection': 'danger',
    'Leukemia': 'danger',
    'Normal_Platelet': 'success',
    'Normal_RBC': 'success',
    'Normal_WBC': 'success',
    'Sickle_Cell_Anemia': 'danger'
}


def load_model(model_path: str) -> BloodClassification:
    """Load the trained PyTorch model from disk."""
    model = BloodClassification()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model


def predict(model: BloodClassification, features: np.ndarray) -> dict:
    """
    Run inference on preprocessed/scaled feature array.
    Returns dict with predicted class, label, confidence, and all probabilities.
    """
    input_tensor = torch.tensor(features, dtype=torch.float32)
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(0)

    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()

    label = DISEASE_CATEGORIES[predicted_class]
    label_tr = DISEASE_CATEGORIES_TR[label]
    severity = DISEASE_SEVERITY[label]

    all_probs = {
        DISEASE_CATEGORIES[i]: round(probs[0][i].item() * 100, 2)
        for i in range(len(DISEASE_CATEGORIES))
    }

    return {
        'predicted_class': predicted_class,
        'label': label,
        'label_tr': label_tr,
        'severity': severity,
        'confidence': round(confidence * 100, 2),
        'probabilities': all_probs
    }
