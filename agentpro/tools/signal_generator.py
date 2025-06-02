from agentpro.tools import Tool
from typing import Dict, Union, List, Any
import numpy as np
from scipy.signal import chirp
import matplotlib.pyplot as plt
from scipy.io import wavfile

class SignalGeneratorAgent(Tool):
    name: str = "Signal Generator Tool"
    description: str = "Generates synthetic signals (sine waves, noise, chirps, multi-tone)."
    action_type: str = "generate_signal"
    input_format: str = """{
        'signal_type': 'sine' | 'noise' | 'chirp' | 'multi_tone' | 'square' | 'noisy_sine',
        'params': {
            'frequency': float (Hz),               # Required for sine, chirp, square
            'amplitude': float,                    # Amplitude of the signal (default=1.0)
            'duration': float (seconds),           # Duration of the signal (default=1.0)
            'sample_rate': int (Hz),              # Sampling rate (default=44100)
            'start_freq': float (Hz),             # For chirp: start frequency
            'end_freq': float (Hz),               # For chirp: end frequency
            'method': 'linear' | 'quadratic'       # For chirp: sweep method
            'frequencies': List[float],            # For multi_tone: list of frequencies
            'amplitudes': List[float]              # For multi_tone: list of amplitudes
        }
    }"""

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Union[np.ndarray, str]]:
        signal_type = input_data.get("signal_type", "sine")
        params = input_data.get("params", {})

        # Generate the signal
        if signal_type == "sine":
            signal = self._generate_sine_wave(**params)
        elif signal_type == "noise":
            signal = self._generate_noise(**params)
        elif signal_type == "chirp":
            signal = self._generate_chirp(**params)
        elif signal_type == "multi_tone":
            signal = self._generate_multi_tone(**params)
        elif signal_type == "square":
            signal = self._generate_square_wave(**params)
        elif signal_type == "noisy_sine":
            signal = self.generate_noisy_signal(**params)
        else:
            raise ValueError(f"Unsupported signal type: {signal_type}")

        return {
            "signal": signal,
            "time_axis": np.linspace(0, params.get("duration", 1.0), len(signal)),
            "message": f"Generated {signal_type} signal."
        }

    # Helper methods for each signal type
    def _generate_sine_wave(
        self,
        frequency: float,
        amplitude: float = 1.0,
        duration: float = 1.0,
        sample_rate: int = 44100,
    ) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        return amplitude * np.sin(2 * np.pi * frequency * t)

    def _generate_noise(
        self,
        amplitude: float = 1.0,
        duration: float = 1.0,
        sample_rate: int = 44100,
    ) -> np.ndarray:
        return amplitude * np.random.normal(0, 1, int(sample_rate * duration))

    def _generate_chirp(
        self,
        start_freq: float,
        end_freq: float,
        duration: float = 1.0,
        method: str = "linear",
        sample_rate: int = 44100,
    ) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        return chirp(t, f0=start_freq, f1=end_freq, t1=duration, method=method)

    def _generate_multi_tone(
        self,
        frequencies: List[float],
        amplitudes: List[float],
        duration: float = 1.0,
        sample_rate: int = 44100,
    ) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = np.zeros_like(t)
        for freq, amp in zip(frequencies, amplitudes):
            signal += amp * np.sin(2 * np.pi * freq * t)
        return signal

    def generate_noisy_signal(
        self,
        frequency: float = 10.0,
        amplitude: float = 1.0,
        duration: float = 1.0,
        sample_rate: int = 44100,
        noise_level: float = 0.3
      ) -> np.ndarray:
        """
        Generate a sine wave with added Gaussian noise.

        Args:
            frequency (float): Frequency of the sine wave in Hz.
            amplitude (float): Amplitude of the sine wave.
            duration (float): Duration of the signal in seconds.
            sample_rate (int): Sampling rate in Hz.
            noise_level (float): Standard deviation of the noise.

        Returns:
            np.ndarray: The noisy signal.
        """
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        sine_wave = amplitude * np.sin(2 * np.pi * frequency * t)
        noise = np.random.normal(0, noise_level, sine_wave.shape)
        noisy_signal = sine_wave + noise
        return noisy_signal

    def _generate_square_wave(
        self,
        frequency: float,
        amplitude: float = 1.0,
        duration: float = 1.0,
        sample_rate: int = 44100,
    ) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))

    # Optional: Save signal as WAV file
    def save_as_wav(self, signal: np.ndarray, filename: str, sample_rate: int = 44100):
        wavfile.write(filename, sample_rate, np.int16(signal * 32767))