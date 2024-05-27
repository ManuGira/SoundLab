import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.io import wavfile
import scipy.signal
import sklearn.linear_model


def plt_spectrogram(fr, ts, Zxx):
    Sxx = abs(Zxx)**2
    plt.pcolormesh(ts, fr, np.log2(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


def herz2octave(hz):
    C3 = 440 * 2 ** (-9 / 12)
    offset = 3 - np.log2(C3)
    res = np.log2(hz) + offset
    res[res < -1] = -1
    res[np.isinf(res)] = -1
    return res


def db2volt(db):
    return 10**(np.array(db).clip(min=-99)/20)


def volt2db(v):
    v = np.array(v)
    v[np.isinf(v)] = db2volt(-99)
    db = 20 * np.log10(v)
    return db.clip(min=-99)


def octave2herz(oct):
    C3 = 440 * 2 ** (-9 / 12)
    offset = 3 - np.log2(C3)
    return 2**(oct - offset)


def plt_stft(f, t, Zxx):
    print("PLOT STFT")

    t1 = t[::10]
    C3 = 440*2**(-9/12)
    octave = np.log2(f/C3)+4
    octave[np.isinf(octave)] = -1
    Zxx1 = abs(Zxx[:, ::10])

    plt.pcolormesh(t1, octave, Zxx1, shading='gouraud')
    plt.ylabel('Frequency [Octaves]')
    plt.xlabel('Time [sec]')
    plt.show()


def linreg_ransac_1d(X1d, y, RESIDUAL_THRESHOLD=None, show_plot=True, plot_xlabel="", plot_ylabel=""):
    assert len(X1d.shape) == 1, "Support only 1 dimensional input"
    X = X1d.reshape((-1,1))
    ransac = sklearn.linear_model.RANSACRegressor(residual_threshold=RESIDUAL_THRESHOLD)
    ransac.fit(X, y)
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    coef_a, coef_b = ransac.estimator_.coef_[0], ransac.estimator_.intercept_
    fitted_line_xs = np.array([min(X1d), max(X1d)])
    fitted_line_ys = coef_a * fitted_line_xs + coef_b
    # PLOT RANSAC RESULT
    # The plot helps to see if outliers have corerlty been rejected
    if show_plot:
        plt.plot(X1d[outlier_mask], y[outlier_mask], 'r.')
        plt.plot(X1d[inlier_mask], y[inlier_mask], 'b.')
        plt.plot(fitted_line_xs, fitted_line_ys, 'g')
        plt.grid()
        plt.title("RANSAC Linear Regression")
        plt.xlabel(plot_xlabel)
        plt.ylabel(plot_ylabel)
        plt.show()
    return coef_a, coef_b, inlier_mask


def compute_db_from_spectrogram(fr_zxx, ts_zxx, Sxx, freqs0, ts0, show_plot=True):
    delta_ts = ts_zxx[-1] - ts_zxx[0]
    Nm1_ts = len(ts_zxx) - 1
    xs = Nm1_ts * (ts0 - ts_zxx[0]) / delta_ts
    delta_fr = fr_zxx[-1] - fr_zxx[0]
    Nm1_fr = len(fr_zxx) - 1
    ys = Nm1_fr * (freqs0 - fr_zxx[0]) / delta_fr
    vol = cv2.remap(Sxx, xs.astype(np.float32), ys.astype(np.float32), interpolation=cv2.INTER_LANCZOS4)
    vol[vol == 0] = -np.inf
    volume = vol / np.median(vol)
    # PLOT SPECTRUM OF SPEAKER
    # convert intensity in db
    db = volt2db(volume)
    if show_plot:
        plt.plot(herz2octave(freqs0), db)
        plt.grid()
        plt.xlabel('Frequency [octaves]')
        plt.ylabel('dB')
        plt.show()
    return db

def get_istft(fr_zxx, ts_zxx, Zxx, Sxx, fs, coef_a, coef_b):
    N = 1024
    freqs0 = np.linspace(fs//N, fs, N)
    octaves0 = herz2octave(freqs0)
    ts0 = (octaves0 - coef_b) / coef_a  # x = (y-b)/a

    delta_ts = ts_zxx[-1] - ts_zxx[0]
    Nm1_ts = len(ts_zxx) - 1
    xs = Nm1_ts * (ts0 - ts_zxx[0]) / delta_ts
    delta_fr = fr_zxx[-1] - fr_zxx[0]
    Nm1_fr = len(fr_zxx) - 1
    ys = Nm1_fr * (freqs0 - fr_zxx[0]) / delta_fr

    # I guess I should use the complex version because the phase is necessary for reconstructing time signal.
    if False:
        volr = cv2.remap(Zxx.real, xs.astype(np.float32), ys.astype(np.float32), interpolation=cv2.INTER_LANCZOS4)
        voli = cv2.remap(Zxx.imag, xs.astype(np.float32), ys.astype(np.float32), interpolation=cv2.INTER_LANCZOS4)
        vol = volr + complex(0, 1)*voli
    else:
        vol = cv2.remap(Sxx, xs.astype(np.float32), ys.astype(np.float32), interpolation=cv2.INTER_LANCZOS4)

    vol = abs(vol) / np.median(abs(vol))
    vol = np.concatenate((vol, vol[::-1]), axis=0).reshape((-1,))

    # no idea if it's actually a good idea to get magnitude only
    res = scipy.fft.ifft(vol, fs)
    res = abs(res[len(res)//2])
    
    plt.plot(res)
    plt.show()

    # vol[vol == 0] = -np.inf
    # db = volt2db(volume)


def fit_sweep(x, fs):
    # COMPUTE SHORT-TERM FOURIER TRANSFORM
    fr_zxx, ts_zxx, Zxx = scipy.signal.stft(x, fs=fs, window='hann', nperseg=2048, boundary='zeros', padded=True, axis=-1)
    Sxx = abs(Zxx)**2

    # PLOT SPECTROGRAM TO VISUALLY APRECIATE THE CHIRP SOUND
    # plt_spectrogram(fr[::10], ts[::10], Zxx[::10, ::10])

    # SELECT POINTS ON THE STFT CORRESPONDING TO THE CHIRP (LOT OF OUTLIER)
    # For each STFT slice, find the maximum instensity.
    # We obtains a list of points in space (Time, Freq) following the chirp sound
    maxs = np.max(abs(Zxx), axis=0)
    inds = np.argmax(abs(Zxx), axis=0)
    freqs = fr_zxx[inds]
    octaves = herz2octave(freqs)  # octaves is the log-scale of frequency

    # DISCARD OULIER USING A RANSAC LINEAR REGRESSION
    # The chirp in the sound file increases its frequency exponentially along time. So we can use the octave scale to make it linear.
    # We simply need to fit a straight line which is pretty easy.
    # We use RANSAC to discard outliers.
    coef_a, coef_b, inlier_mask = linreg_ransac_1d(
        X1d=ts_zxx,
        y=octaves,
        RESIDUAL_THRESHOLD=0.03,  # Totally heuristic threshold based on my data. The lower the less inliers.
        show_plot=True,
        plot_xlabel='Time [sec]',
        plot_ylabel='Frequency [Octaves]',
    )

    # READ INTENSITIES ON SPECTROGRAM
    # OCTAVES vs DB
    octaves0 = np.linspace(0, 10, 1001)
    ts0 = (octaves0 - coef_b) / coef_a  # x = (y-b)/a
    freqs0 = octave2herz(octaves0)
    db = compute_db_from_spectrogram(fr_zxx, ts_zxx, Sxx, freqs0, ts0, show_plot=True)

    # FREQS vs DB
    ts0 = (octaves0 - coef_b) / coef_a  # x = (y-b)/a
    freqs0 = np.linspace(10, 20000, 1001)
    db = compute_db_from_spectrogram(fr_zxx, ts_zxx, Sxx, freqs0, ts0, show_plot=True)
    volume = db2volt(db)

    # inverse STFT
    get_istft(fr_zxx, ts_zxx, Zxx, Sxx, fs, coef_a, coef_b)



def main():
    # src = "inputs/Chirp HP Omen.wav"
    # src = "inputs/Chirp HiFi.wav"
    src = "inputs/Chirp Beoplay.wav"

    fs, data_int16 = wavfile.read(src)
    data = data_int16.copy() / (2 ** 15 - 1)

    fit_sweep(data, fs)


if __name__ == '__main__':
    main()