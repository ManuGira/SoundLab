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


def generate_risset_beat(sound_array, repetitions=4, octaves=3):
    """
    time speed goes from x1 to x2
    """

    samples = []
    samples.append(sound_array.copy())
    samples[0] = np.tile(samples[0], (repetitions, 1))
    for i in range(1, octaves):
        new_octave = double_time_speed(samples[i-1])
        samples.append(new_octave)

    # define time remapping
    N = len(samples[0])
    ns = time_map_eponentially_accelerated(N, 2)
    M = len(ns)

    # nearest neighbor
    ns = np.round(ns).astype(int)

    # resample (remap time)
    for i in range(octaves):
        samples[i] = samples[i][ns]

    # fade in/out first and last layers
    fade_in = np.linspace(0, 1, M)
    fade_out = np.linspace(1, 0, M)
    fade_in = fade_in.reshape(-1, 1)
    fade_out = fade_out.reshape(-1, 1)
    samples[0] = samples[0]*fade_in
    samples[-1] = samples[-1]*fade_out

    # mix layers and normalize to [-1, 1]
    samples = sum(samples)
    max_amp = np.max(np.abs(samples.flatten()))
    samples /= max_amp
    return samples


def main():
    filename = "indus lofi.wav"
    fs, data_int16 = wavfile.read(f"assets/{filename}")
    data = data_int16.copy() / (2 ** 15 - 1)
    # gui(data)
    data = np.repeat(data, 2, axis=0)
    samples = generate_risset_beat(data, repetitions=2, octaves=3)

    samples = np.tile(samples, (4, 1))
    wavfile.write(f"outputs/{filename}", fs, samples)

if __name__ == '__main__':
    main()