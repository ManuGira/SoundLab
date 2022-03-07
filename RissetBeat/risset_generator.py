from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

def risset_generator2(sound_array, repetition):
    """
    time speed x2
    """
    # define time remapping
    N = len(sound_array)
    ln2 = np.log(2)
    M = int(ln2*N+1)
    ns = np.arange(0, M)
    ns = (np.exp(ns/N)-1)*N

    # nearest neighbor
    ns = np.round(ns).astype(int)

    # resample (remap time)
    samples = sound_array[ns]
    return samples


def double_time_speed(sound_array):
    samples = sound_array.copy()
    N = len(sound_array)
    ns0 = np.arange(start=0, stop=N, step=2)
    ns1 = np.arange(start=1, stop=N, step=2)
    ns = np.concatenate((ns0, ns1))

    samples = samples[ns]
    return samples


def time_map_eponentially_accelerated(N, final_speed):
    lnf = np.log(final_speed)
    M = int(N*lnf/(final_speed - 1))
    ns = np.arange(M)
    ns = (np.exp(ns*lnf/M) - 1) * M/lnf
    return ns


def generate_risset_beat(sound_array, repetitions=4):
    """
    time speed goes from x1 to x2
    """
    samples1 = sound_array.copy()
    samples1 = np.tile(samples1, (repetitions, 1))
    samples2 = double_time_speed(samples1)

    # define time remapping
    N = len(samples1)
    ns = time_map_eponentially_accelerated(N, 2)
    M = len(ns)

    # nearest neighbor
    ns = np.round(ns).astype(int)

    # resample (remap time)
    samples1 = samples1[ns]
    samples2 = samples2[ns]

    # fade in/out first and last layers
    fade_in = np.linspace(0, 1, M)
    fade_out = np.linspace(1, 0, M)
    fade_in = fade_in.reshape(-1, 1)
    fade_out = fade_out.reshape(-1, 1)
    samples1 = samples1*fade_in
    samples2 = samples2*fade_out

    # mix layers and normalize to [-1, 1]
    samples = samples1 + samples2
    max_amp = np.max(np.abs(samples.flatten()))
    samples /= max_amp
    return samples


def main():
    fs, data_int16 = wavfile.read("assets/Risset2.wav")
    data = data_int16.copy() / (2 ** 15 - 1)
    samples = generate_risset_beat(data)

    wavfile.write("outputs/risset.wav", fs, samples)

if __name__ == '__main__':
    main()