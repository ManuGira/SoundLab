import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt


def display_risset_spiral(sound_array, repetitions, octaves):
    samples = sound_array.copy()

    # resample (remap time)
    time_mesh, time_speed, loop_counts = compute_time_mesh(len(samples), repetitions, octaves)
    samples = samples[time_mesh]
    M = len(samples)//octaves
    samples = apply_fade_window(samples, M)

    # down sample to lighten CPU
    N0 = len(samples)
    N = 3000*repetitions
    ns = np.linspace(start=0, stop=N0-1, num=N, dtype=int)
    samples = samples[ns]
    time_speed = time_speed[ns]
    loop_counts = loop_counts[ns]

    # convert stereo to mono
    samples = np.sum(samples, axis=1)

    nb_points = len(samples)
    rho = time_speed+samples
    phi = np.linspace(0, octaves*2*np.pi, nb_points)

    # xs = rho * np.cos(phi)
    # ys = rho * np.sin(phi)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    # ax.plot(phi, rho)
    for i in range(loop_counts[-1]+1):
        color = "b" if i%2==0 else "r"
        ax.plot(phi[loop_counts == i], rho[loop_counts == i], color=color)
    plt.show()


def time_map_eponentially_accelerated(N, speed_factor_per_octaves, octaves):
    lnsf = np.log(speed_factor_per_octaves)
    final_speed = speed_factor_per_octaves ** (octaves + 1)
    nb_laps = np.log(final_speed / speed_factor_per_octaves) / lnsf
    M = int(N * lnsf / (speed_factor_per_octaves - 1))
    ns = np.arange(M*nb_laps)
    ns = (np.exp(ns*lnsf/M) - 1) * M/lnsf
    return ns


def compute_time_mesh(sample_length, repetitions, octaves):
    speed_factor = 2

    time_mesh = time_map_eponentially_accelerated(repetitions * sample_length, speed_factor_per_octaves=speed_factor, octaves=octaves)  # todo

    # compute derivative for display
    time_speed = np.ones_like(time_mesh)
    time_speed[1:] = time_mesh[1:] - time_mesh[:-1]

    # compute loop nb
    loop_count = (time_mesh/sample_length).astype(int)

    # nearest neighbor interpolation
    time_mesh = np.round(time_mesh).astype(int)


    # remainder to make sure no value is >= N
    time_mesh = np.remainder(time_mesh, sample_length)


    return time_mesh, time_speed, loop_count


def apply_fade_window(samples, fade_length):
    # fade in/out
    samples = samples.copy()
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)
    fade_in = fade_in.reshape(-1, 1)
    fade_out = fade_out.reshape(-1, 1)
    samples[:fade_length] = samples[:fade_length] * fade_in
    samples[-fade_length:] = samples[-fade_length:] * fade_out
    return samples


def generate_risset_beat(sound_array, repetitions=4, octaves=3):
    samples = sound_array.copy()

    # resample (remap time)
    time_mesh, time_speed, loop_counts = compute_time_mesh(len(samples), repetitions, octaves)
    samples = samples[time_mesh]

    M = len(samples)//octaves
    samples = apply_fade_window(samples, M)


    # fold spiral on itself and sum up to mix the different octaves
    _, D = samples.shape
    samples = samples.reshape(octaves, -1, D)
    samples = np.sum(samples, axis=0)

    # normalize to [-1, 1]
    max_amp = np.max(np.abs(samples.flatten()))
    samples /= max_amp
    return samples


def main():
    src = "assets/example.wav"
    repetitions = 1
    octaves = 4

    fs, data_int16 = wavfile.read(src)
    data = data_int16.copy() / (2 ** 15 - 1)

    # half time
    # data = np.repeat(data, 2, axis=0)

    samples = generate_risset_beat(data, repetitions, octaves)

    # repeat x4 and save
    samples = np.tile(samples, (4, 1))
    filename = os.path.basename(src)
    ext = filename.split(".")[-1]
    filename = filename[:-len(ext)-1]
    dst = f"outputs/{filename} - RissetBeat_R{repetitions}_O{octaves}.{ext}"

    samples_int16 = (samples*(2**15-1)).astype(np.int16)
    wavfile.write(dst, fs, samples_int16)
    display_risset_spiral(data, repetitions, octaves)


if __name__ == '__main__':
    main()