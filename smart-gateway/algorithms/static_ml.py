from pathlib import Path

import joblib
import numpy as np


class StaticMLRouter:
    def __init__(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"model not found: {path}\n"
                "Train it first: cd ml-training && python generate_dataset.py && python train_model.py"
            )
        self.model = joblib.load(path)

    def predict(self, prompt: str, gpu_in_flight: int, cpu_in_flight: int) -> int:
        features = self._extract_features(prompt, gpu_in_flight, cpu_in_flight)
        return int(self.model.predict([features])[0])

    def _extract_features(
        self, prompt: str, gpu_in_flight: int, cpu_in_flight: int
    ) -> np.ndarray:
        prompt_length = len(prompt.split())
        return np.array([prompt_length, gpu_in_flight, cpu_in_flight], dtype=float)
