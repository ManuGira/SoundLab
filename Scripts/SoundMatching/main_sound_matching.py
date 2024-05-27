from soundlab import tools
from soundlab import spectrogram
import numpy as np
import guibbon as gbn
import cv2
import matplotlib.pyplot as plt


def load_fingerprint(wav_filename: str, stft_encoder: spectrogram.IEncoder):
    wav = tools.read_wav(wav_filename, target_fs=stft_encoder.fs, target_win_size=stft_encoder.win_size)
    fr, ts, S = stft_encoder.encode(wav)
    fingerprint = abs(S)
    fingerprint = (np.clip(fingerprint, 0, 1)*255).astype(np.uint8)
    return fr, ts, fingerprint


def main():
    FR_MAX = 5000

    FS = FR_MAX * 2
    WIN_SIZE = 2 ** (int(1 + np.log2(FS / 60)))
    scipy_stft_encoder = spectrogram.ScipySTFTEncoder(FS, WIN_SIZE)

    fr_a, ts_a, fingerprint_a = load_fingerprint("data/challenge_level_1.wav", scipy_stft_encoder)
    fr_p, ts_p, fingerprint_p = load_fingerprint("data/short.wav", scipy_stft_encoder)

    gbn.imshow("", fingerprint_a)
    gbn.waitKeyEx(0)

    res = cv2.matchTemplate(fingerprint_a, fingerprint_p, method=cv2.TM_CCOEFF_NORMED)

    print(res)
    plt.plot(res[0, :])
    plt.show()

if __name__ == '__main__':
    main()
