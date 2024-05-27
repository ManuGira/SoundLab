import abc

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hann

WIN_SIZE = 2048
FS = 48000


def plt_spectrogram(fr, ts, Zxx):
    Sxx = abs(Zxx) ** 2
    plt.pcolormesh(ts, fr, np.log2(Sxx), shading="gouraud")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.show()


def plt_complex_array2d(Z):
    magn = abs(Z)
    phase = np.angle(Z)

    height, width = Z.shape

    hue = (phase * 255 / (2 * np.pi)).astype(np.uint8).reshape(height, width, 1)
    light = (np.log(magn + 1) * 200).astype(np.uint8).reshape(height, width, 1)

    hls = np.concatenate(
        [
            hue,
            light,
            np.zeros_like(hue) + 127,
        ],
        axis=2,
    )

    cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)
    cv2.imshow("complex array 2d", hls)
    cv2.waitKey(0)


class IEncoder(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def encode(self, y):
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, S):
        raise NotImplementedError


class ScipySTFTEncoder(IEncoder):
    def __init__(self, fs, win_size):
        self.fs = fs
        self.win_size = win_size

        # win = gaussian(self.win_size, std=self.win_size // 6, sym=True)  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.gaussian.html#scipy.signal.windows.gaussian
        win = hann(self.win_size, sym=True)  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.hann.html

        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html
        self.SFT = ShortTimeFFT(win, hop=self.win_size // 2, fs=self.fs, scale_to=None, fft_mode="onesided")

    def encode(self, y):
        # convert stereo to mono if needed
        if len(y.shape) > 1:
            y = y[:, 0]

        # COMPUTE SHORT-TERM FOURIER TRANSFORM
        S = self.SFT.stft(y)

        n = S.shape[1]
        ts = np.arange(n) * self.SFT.delta_t
        fr = self.SFT.f
        return fr, ts, S

    def decode(self, S):
        return self.SFT.istft(S)


class LibrosaSTFTEncoder(IEncoder):
    def __init__(self, fs, win_size):
        self.fs = fs
        self.win_size = win_size

        self.window = "hann"
        self.center = True
        self.hop_length = self.win_size // 2
        self.delta_t = self.hop_length / self.fs
        self.delta_f = self.fs / self.win_size

    def encode(self, y):
        S = librosa.stft(
            y=y,
            n_fft=self.win_size,
            hop_length=self.hop_length,
            win_length=self.win_size,
            window=self.window,
            center=True,
        )
        fr = np.arange(1 + self.win_size / 2) * self.delta_f
        ts = np.arange(S.shape[-1]) * self.delta_t
        return fr, ts, S

    def decode(self, S):
        # destroy and recover phase using Griffin Lim algorithm
        y = librosa.griffinlim(
            abs(S),  # use abs to keep magnitude only and drop phase
            n_fft=self.win_size,
            hop_length=self.win_size // 2,
            win_length=self.win_size,
            window=self.window,
            center=True,
        )
        return y


def main():
    dataset_raw = "../dataset/raw/"
    dataset_formatted = "../dataset/formatted/"
    output_folder = "../dataset/output/"

    filename = "cw_amen03_167.wav"
    data = read_wav(dataset_raw + filename, target_fs=FS, target_win_size=WIN_SIZE)

    # tools.write_wav(dataset_formatted + filename, data, target_fs=FS)

    # tools.write_wav(f"{output_folder}{filename[:-4]}_sawtooth_clean.wav", data, target_fs=FS)

    scipy_stft_encoder = ScipySTFTEncoder(FS, WIN_SIZE)
    librosa_stft_encoder = LibrosaSTFTEncoder(FS, WIN_SIZE)

    # Encode to stft. Scipy is faster than librosa
    fr, ts, S = scipy_stft_encoder.encode(data)

    # clear phase (destructive operation)
    S = abs(S)

    # Librosa is able to perform inverse-STFT with phase reconstruction thanks to Griffin Lim algo.
    data_out = librosa_stft_encoder.decode(S)

    # tools.write_wav(f"{output_folder}{filename[:-4]}_reconstructed.wav", data_out, FS)


if __name__ == "__main__":
    main()
