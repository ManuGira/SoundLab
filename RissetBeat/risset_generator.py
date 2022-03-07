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
    samples = sound_array.copy()
    N, D = samples.shape
    meshes = []
    speed_factor = 2
    ns = time_map_eponentially_accelerated(repetitions*N, final_speed=speed_factor)
    M = len(ns)
    ns = np.remainder(ns, N)
    meshes.append(ns)
    for i in range(1, octaves):
        meshes.append(np.remainder(ns*speed_factor**i, N))
    time_mesh = np.concatenate(meshes, axis=0)

    # nearest neighbor interpolation
    time_mesh = np.round(time_mesh).astype(int)
    # remainder to make sure no value is >= N
    time_mesh = np.remainder(time_mesh, N)


    # resample (remap time)
    samples = samples[time_mesh]

    # fade in/out
    fade_in = np.linspace(0, 1, M)
    fade_out = np.linspace(1, 0, M)
    fade_in = fade_in.reshape(-1, 1)
    fade_out = fade_out.reshape(-1, 1)
    samples[:M] = samples[:M]*fade_in
    samples[-M:] = samples[-M:]*fade_out

    # fold spiral on itself and sum up to mix the different octaves
    samples = samples.reshape(octaves, -1, D)
    samples = np.sum(samples, axis=0)

    # normalize to [-1, 1]
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