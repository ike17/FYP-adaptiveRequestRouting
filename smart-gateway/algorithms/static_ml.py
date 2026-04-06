# smart-gateway/algorithms/static_ml.py
# Wraps the pre-trained scikit-learn Random Forest for request-level routing.
#
# Feature space (3 features — all observable at gateway request time):
#   prompt_length  : word count of the full augmented prompt (after RAG context injection)
#   gpu_in_flight  : current number of requests in-flight to the GPU Ollama
#   cpu_in_flight  : current number of requests in-flight to the CPU Ollama
#
# Label: 0 = route to GPU, 1 = route to CPU
#
# Train the model with: cd ml-training && python generate_dataset.py && python train_model.py

from pathlib import Path

import joblib
import numpy as np


class StaticMLRouter:
    """Loads a pre-trained Random Forest and predicts the target node per request."""

    def __init__(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"model not found: {path}\n"
                "Train it first: cd ml-training && python generate_dataset.py && python train_model.py"
            )
        self.model = joblib.load(path)

    def predict(self, prompt: str, gpu_in_flight: int, cpu_in_flight: int) -> int:
        """Return 0 (GPU) or 1 (CPU)."""
        features = self._extract_features(prompt, gpu_in_flight, cpu_in_flight)
        return int(self.model.predict([features])[0])

    def _extract_features(
        self, prompt: str, gpu_in_flight: int, cpu_in_flight: int
    ) -> np.ndarray:
        prompt_length = len(prompt.split())
        return np.array([prompt_length, gpu_in_flight, cpu_in_flight], dtype=float)
