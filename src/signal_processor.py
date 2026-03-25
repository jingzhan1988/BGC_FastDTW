"""
信号预处理模块
Signal Processor Module

包含信号滤波、包络计算、波形检测等功能
"""

import numpy as np
import pywt
from scipy.signal import find_peaks, hilbert, resample
from typing import List, Tuple, Optional

from .config import Config, get_config


class SignalProcessor:
    """信号预处理器"""

    def __init__(self, config: Config = None):
        """
        初始化信号处理器

        Args:
            config: 配置对象
        """
        self.config = config or get_config()
        self.signal_config = self.config.signal

    def smart_psd(self, x: np.ndarray, nfft: int) -> np.ndarray:
        """
        周期图法频谱估计

        Args:
            x: 输入信号
            nfft: FFT点数

        Returns:
            功率谱密度
        """
        N_sample = nfft
        Fs = self.signal_config.target_sample_rate

        # 计算FFT
        x_FFT = np.abs(np.fft.fft(x, N_sample)) / N_sample * 2
        x_FFT[0] /= 4  # 直流分量
        x_FFT[N_sample // 2] /= 8  # 奈奎斯特频率

        FFT_PSD = (x_FFT ** 2) / 2
        FFT_PSD[0] = x_FFT[0] ** 2
        FFT_PSD[N_sample // 2] = x_FFT[N_sample // 2] ** 2
        FFT_PSD = FFT_PSD / Fs * N_sample

        return FFT_PSD

    def energy_envelope(self, x: np.ndarray) -> np.ndarray:
        """
        计算信号的能量包络

        Args:
            x: 输入信号

        Returns:
            能量包络
        """
        window_size = self.signal_config.envelope_window_size
        frame = len(x) // 100
        M = frame * 100
        envelope = np.zeros(M)

        for n in range(M):
            end_idx = min(n + window_size, len(x))
            envelope[n] = np.sqrt(np.sum(x[n:end_idx] ** 2))

        return envelope

    def wavelet_filter(self, signal: np.ndarray) -> np.ndarray:
        """
        小波滤波

        Args:
            signal: 输入信号

        Returns:
            滤波后的信号
        """
        wavelet = self.signal_config.wavelet
        level = self.signal_config.wavelet_level
        detail_levels = self.signal_config.wavelet_detail_levels

        # 小波分解
        wt = pywt.wavedec(signal, wavelet, level=level)

        # 取指定层的细节系数
        min_level = min(detail_levels) - 1  # 转换为0索引
        filtered = sum(wt[min_level:])

        return filtered

    def hilbert_envelope(self, signal: np.ndarray) -> np.ndarray:
        """
        Hilbert变换计算包络

        Args:
            signal: 输入信号

        Returns:
            包络信号
        """
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        envelope -= np.mean(envelope)
        return envelope

    def smooth_envelope(self, envelope: np.ndarray, iterations: int = None) -> np.ndarray:
        """
        平滑包络

        Args:
            envelope: 包络信号
            iterations: 平滑迭代次数

        Returns:
            平滑后的包络
        """
        if iterations is None:
            iterations = self.signal_config.envelope_smooth_iterations

        window_size = self.signal_config.envelope_window_size
        kernel = np.ones(window_size) / window_size

        result = envelope
        for _ in range(iterations):
            result = np.convolve(result, kernel, mode='same')

        return result

    def find_envelope_peaks(self, envelope: np.ndarray, signal: np.ndarray,
                            freq_temp: float, frame: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据包络峰值定位信号峰值

        Args:
            envelope: 包络信号
            signal: 原始信号
            freq_temp: 频率阈值
            frame: 帧长度

        Returns:
            (peak_positions, peak_values)
        """
        e_peak, _ = find_peaks(envelope, height=freq_temp)
        x_peak = []
        x_peak_val = []

        for n in range(len(e_peak)):
            if n == 0:
                nt = e_peak[n]
                x_temp = signal[:nt + 2]
                tt = find_peaks(np.diff(np.sign(np.diff(x_temp))))[0] + 1

                if len(tt) == 0:
                    peak_val = signal[nt]
                    x_peak.append(nt)
                    x_peak_val.append(peak_val)
                else:
                    peak = np.argmax(x_temp[tt])
                    peak_val = signal[peak]
                    x_peak.append(peak)
                    x_peak_val.append(peak_val)
            else:
                if e_peak[n] < frame * 100 - 2:
                    x_temp = signal[e_peak[n - 1] + 1:e_peak[n]]
                    tt = find_peaks(np.diff(np.sign(np.diff(x_temp))))[0] + 1

                    if len(tt) == 0:
                        x_peak.append(e_peak[n])
                        x_peak_val.append(signal[e_peak[n]])
                    else:
                        peak = np.argmax(x_temp[tt]) + e_peak[n - 1] + 1
                        x_peak.append(peak)
                        peak_val = signal[min(peak + 1, len(signal) - 1)]
                        x_peak_val.append(peak_val)

        return np.array(x_peak), np.array(x_peak_val)

    def find_wave_positions(self, signal: np.ndarray, peak_positions: np.ndarray) -> dict:
        """
        定位BCG波形的各个子波位置 (H, I, J, K, L, M, N)

        Args:
            signal: 输入信号
            peak_positions: 主峰位置

        Returns:
            波形位置字典
        """
        signal = signal.flatten()
        n = len(peak_positions)

        if n < 3:
            return None

        waves = {
            'H': [], 'I': [], 'J': [], 'K': [], 'L': [], 'M': [], 'N': []
        }

        # J波直接使用峰值位置
        for i in range(1, n - 1):
            idx = int(peak_positions[i])
            waves['J'].append({'position': idx, 'value': signal[idx]})

        for ik in range(n - 2):
            try:
                j_pos = int(peak_positions[ik + 1])
                x_before = signal[:j_pos]
                x_after = signal[j_pos:]

                # H波 - J波前的波峰
                tt = find_peaks(x_before)[0]
                if len(tt) > 0:
                    th = tt[-1]
                    waves['H'].append({'position': th, 'value': signal[th]})
                else:
                    waves['H'].append(None)

                # I波 - J波前的波谷
                tt = find_peaks(-x_before)[0]
                if len(tt) > 0:
                    ti = tt[-1]
                    waves['I'].append({'position': ti, 'value': signal[ti]})
                else:
                    waves['I'].append(None)

                # K波 - J波后的波谷
                tt = find_peaks(-x_after)[0]
                if len(tt) > 0:
                    tk = tt[0] + j_pos
                    waves['K'].append({'position': tk, 'value': signal[tk]})
                else:
                    waves['K'].append(None)

                # L波 - K波后的波峰
                tt = find_peaks(x_after)[0]
                if len(tt) > 0:
                    tl = tt[0] + j_pos
                    waves['L'].append({'position': tl, 'value': signal[tl]})
                else:
                    waves['L'].append(None)

                # M波 - L波后的波谷
                tt = find_peaks(-x_after)[0]
                if len(tt) > 1:
                    tm = tt[1] + j_pos
                    waves['M'].append({'position': tm, 'value': signal[tm]})
                else:
                    waves['M'].append(None)

                # N波 - M波后的波峰
                tt = find_peaks(x_after)[0]
                if len(tt) > 1:
                    tn = tt[1] + j_pos
                    waves['N'].append({'position': tn, 'value': signal[tn]})
                else:
                    waves['N'].append(None)

            except Exception:
                for key in ['H', 'I', 'K', 'L', 'M', 'N']:
                    waves[key].append(None)

        return waves

    def find_n_wave_position(self, signal: np.ndarray) -> Optional[int]:
        """
        快速检测N波位置

        Args:
            signal: 输入信号

        Returns:
            N波位置索引
        """
        signal = np.asarray(signal).flatten()

        try:
            peaks, _ = find_peaks(signal)

            if len(peaks) < 2:
                return len(signal) - 1

            # 找主峰
            main_peak_idx = peaks[np.argmax(signal[peaks])]

            # N波是主峰后的第二个峰
            peaks_after_main = peaks[peaks > main_peak_idx]

            if len(peaks_after_main) >= 2:
                return int(peaks_after_main[1])
            elif len(peaks_after_main) == 1:
                return int(peaks_after_main[0])
            else:
                return len(signal) - 1

        except Exception:
            return len(signal) - 1

    def truncate_to_n_wave(self, signal: np.ndarray) -> np.ndarray:
        """
        截断信号到N波位置

        Args:
            signal: 输入信号

        Returns:
            截断后的信号
        """
        n_wave_idx = self.find_n_wave_position(signal)
        if n_wave_idx is not None and n_wave_idx < len(signal):
            return signal[:n_wave_idx + 1]
        return signal

    def preprocess(self, signal: np.ndarray, target_length: int = None) -> np.ndarray:
        """
        完整的信号预处理流程

        Args:
            signal: 输入信号
            target_length: 目标长度（用于重采样）

        Returns:
            预处理后的信号
        """
        signal = np.asarray(signal).flatten()

        # 重采样
        if target_length is not None:
            signal = resample(signal, target_length)

        # 小波滤波
        filtered = self.wavelet_filter(signal)

        return filtered

    def segment_cardiac_cycles(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        分割心动周期

        Args:
            signal: 输入信号

        Returns:
            心动周期列表
        """
        # 重采样到目标采样率
        target_len = int(len(signal) * self.signal_config.target_sample_rate
                        / self.signal_config.original_sample_rate)
        signal_resampled = resample(signal, target_len)

        # 小波滤波
        filtered = self.wavelet_filter(signal_resampled)

        # 计算包络
        envelope = self.hilbert_envelope(filtered)
        energy_env = self.energy_envelope(filtered)
        smooth_env = self.smooth_envelope(energy_env)

        # 定位峰值
        try:
            # 估计频率阈值
            psd = self.smart_psd(envelope[50:750] if len(envelope) > 750 else envelope, 2048)
            freq_threshold = np.max(psd) * 0.3

            peaks, _ = self.find_envelope_peaks(smooth_env, filtered, freq_threshold)

            if len(peaks) < 2:
                return [filtered]

            # 分割
            cycles = []
            for i in range(len(peaks) - 1):
                start = int(peaks[i])
                end = int(peaks[i + 1])
                if end > start:
                    cycles.append(filtered[start:end])

            return cycles if cycles else [filtered]

        except Exception:
            return [filtered]


# 便捷函数
def preprocess_signal(signal: np.ndarray, config: Config = None) -> np.ndarray:
    """预处理信号的便捷函数"""
    processor = SignalProcessor(config)
    return processor.preprocess(signal)


def find_n_wave(signal: np.ndarray, config: Config = None) -> Optional[int]:
    """检测N波位置的便捷函数"""
    processor = SignalProcessor(config)
    return processor.find_n_wave_position(signal)


def truncate_signal(signal: np.ndarray, config: Config = None) -> np.ndarray:
    """截断信号到N波的便捷函数"""
    processor = SignalProcessor(config)
    return processor.truncate_to_n_wave(signal)
