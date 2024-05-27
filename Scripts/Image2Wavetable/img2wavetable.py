import os
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from scipy.io import wavfile


def display(wave, img):
    imgd = img.copy().astype(np.float32)
    imgd -= imgd.min()
    imgd /= imgd.max()
    imgd = (255*imgd).astype(np.uint8)

    imgd = cv.resize(imgd, dsize=(1024, 510), interpolation=cv.INTER_NEAREST)
    cv.imshow("wind", imgd)

    plt.plot(wave)
    plt.show()
    cv.waitKey(0)

def comb_filter(img, harmonic):
    out = np.zeros_like(img)
    _, W = out.shape
    a = 1/harmonic
    for h in range(harmonic):
        d = (h*W)//harmonic
        delay = np.hstack([img[:, d:], img[:, :d]])
        out += a*delay
        a *= -1
    return out


def playsound_background(src):
    try:
        import playsound
    except ImportError as e:
        print("Please, install playsound to preview result")
        return
    import threading
    soundthread = threading.Thread(target=playsound.playsound, args=(src,), name='backgroundMusicThread')
    soundthread.daemon = True  # shut down music thread when the rest of the program exits
    soundthread.start()


def main():
    src = "inputs/example.jpg"
    harmonic = 2
    equalize_left_right = True

    img = cv.imread(src)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    H = 128
    W = 1024

    img = cv.resize(img, dsize=(W, H), interpolation=cv.INTER_NEAREST)
    img = img.astype(np.float32)

    # equalize left and right sides
    if equalize_left_right:
        for y in range(H):
            left, right = img[y, ::W-1]
            img[y, :] -= np.linspace(left, right, W)

    # comb filter of length W/2 to keep only harmonics
    if harmonic > 0:
        img = comb_filter(img, harmonic)

    # normalize
    # it can be exported if it is in float32 in [-1.0, 1.0] or uint8 in [0, 255]
    img -= img.min()
    img /= img.max()
    img = 2*img-1

    wave = img[::-1, :].flatten(order='C')

    filename = os.path.basename(src)
    ext = filename.split(".")[-1]
    filename = filename[:-len(ext) - 1]
    dst = f"outputs/{filename}_comb{harmonic}{'_EQLF' if equalize_left_right else ''}.wav"
    data_int16 = (wave * (2 ** 15 - 1)).astype(np.int16)
    wavfile.write(dst, rate=48000, data=data_int16)

    playsound_background(dst)
    display(wave, img)

if __name__ == '__main__':
    main()