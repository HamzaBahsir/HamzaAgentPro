from typing import Dict, Any, List, Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.fft import fft
from sklearn.ensemble import IsolationForest

import torch
import torch.nn.functional as F
from torchvision.models import resnet18

from pydantic import ConfigDict
from agentpro.tools import Tool


class SignalAnalyzerAgent(Tool):
    # Tell Pydantic to allow arbitrary (non‐Pydantic) types here:
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Declare fields so Pydantic knows they exist
    default_sample_rate: int = 44100
    anomaly_detector: Optional[IsolationForest] = None
    classifier: Optional[torch.nn.Module] = None

    name: str = "Signal Analyzer Tool"
    description: str = (
        "Performs advanced signal analysis including FFT, STFT, "
        "anomaly detection, and ML classification. "
        "For classification, this tool resizes the spectrogram to 224×224 and "
        "expands it to 3 channels before feeding into ResNet-18."
    )
    action_type: str = "signal_analysis"
    input_format: str = """{
        "signal": List[float],            # 1D input signal array
        "sample_rate": int,               # Sampling rate in Hz
        "analysis_types": List[str]       # Any of: 'fft', 'stft', 'anomaly', 'classification'
    }"""

    def __init__(self):
        super().__init__()
        # Now Pydantic will allow these arbitrary types
        self.anomaly_detector = IsolationForest(contamination=0.05)
        self.classifier = self._load_pretrained_model()

    def _load_pretrained_model(self) -> torch.nn.Module:
        """Load and configure a pretrained ResNet-18 for 5-class output."""
        model = resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 5)  # 5 output classes
        return model.eval()

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for the tool."""
        raw_signal = np.array(input_data["signal"], dtype=np.float32)
        sample_rate = input_data.get("sample_rate", self.default_sample_rate)
        analysis_types = input_data.get("analysis_types", ["fft"])

        results: Dict[str, Any] = {}

        # 1) FFT
        if "fft" in analysis_types:
            freqs, mags = self._compute_fft(raw_signal, sample_rate)
            results["fft"] = {
                "frequencies": freqs.tolist(),
                "magnitudes": mags.tolist()
            }

        # 2) STFT (compute if requested or needed for classification)
        if "stft" in analysis_types or "classification" in analysis_types:
            f, t, Zxx = self._compute_stft(raw_signal, sample_rate)
            results["stft"] = {
                "frequencies": f.tolist(),
                "times": t.tolist(),
                "spectrogram": np.abs(Zxx).tolist()
            }

        # 3) Anomaly Detection
        if "anomaly" in analysis_types:
            anomalies = self._detect_anomalies(raw_signal)
            results["anomalies"] = anomalies.tolist()

        # 4) Classification
        if "classification" in analysis_types:
            # Reuse STFT result if present, otherwise recompute
            if "stft" in results:
                spectrogram = np.array(results["stft"]["spectrogram"], dtype=np.float32)
            else:
                _, _, Zxx = self._compute_stft(raw_signal, sample_rate)
                spectrogram = np.abs(Zxx).astype(np.float32)

            class_id = self._classify_signal(spectrogram)
            results["classification"] = int(class_id)

        return results

    def _compute_fft(self, signal_data: np.ndarray, sample_rate: int):
        """Compute single‐sided FFT magnitude."""
        N = len(signal_data)
        yf = fft(signal_data)
        xf = np.linspace(0, sample_rate / 2, N // 2, endpoint=True)
        mags = (2.0 / N) * np.abs(yf[: N // 2])
        return xf, mags

    def _compute_stft(self, signal_data: np.ndarray, sample_rate: int, nperseg: int = 256):
        """
        Compute STFT. Returns:
          - f: frequency bins
          - t: segment times
          - Zxx: complex STFT matrix (shape = [len(f), len(t)])
        """
        f, t, Zxx = sp_signal.stft(signal_data, fs=sample_rate, nperseg=nperseg)
        return f, t, Zxx

    def _detect_anomalies(self, signal_data: np.ndarray):
        """
        Fit IsolationForest on three summary features:
        [max_value, mean_abs, std_dev].
        Returns array of labels (–1 for outlier, +1 for inlier).
        """
        features = np.array([[
            np.max(signal_data),
            np.mean(np.abs(signal_data)),
            np.std(signal_data)
        ]], dtype=np.float32)  # shape = (1, 3)

        labels = self.anomaly_detector.fit_predict(features)
        return labels

    def sum_signals(sine_wave: np.ndarray, noise: np.ndarray) -> np.ndarray:
        """
        Sum sine wave and noise element-wise for FFT analysis.

        Args:
            sine_wave (np.ndarray): Clean sine wave signal.
            noise (np.ndarray): Noise signal of the same length as sine_wave.

        Returns:
            np.ndarray: Combined signal ready for FFT.
        """
        if sine_wave.shape != noise.shape:
            raise ValueError("Sine wave and noise must have the same shape.")

        combined_signal = sine_wave.astype(np.float32) + noise.astype(np.float32)
        return combined_signal


    def _classify_signal(self, spectrogram: np.ndarray):
        """
        Take a 2D spectrogram (shape [freq_bins, time_steps]) → resize to 224×224
        → expand to 3 channels → feed into ResNet-18.
        Returns the class index (0…4).
        """
        # 1) Convert to torch.Tensor and add channel & batch dims: [1, 1, F, T]
        tensor_spec = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0)  # [1, 1, F, T]

        # 2) Resize to (1, 1, 224, 224):
        tensor_resized = F.interpolate(tensor_spec, size=(224, 224), mode="bilinear", align_corners=False)

        # 3) Squeeze out the single channel, then repeat to 3 channels: [1, 3, 224, 224]
        tensor_rgb = tensor_resized.squeeze(1).repeat(1, 3, 1, 1)

        with torch.no_grad():
            outputs = self.classifier(tensor_rgb)
            predicted = torch.argmax(outputs, dim=1).item()

        return predicted
