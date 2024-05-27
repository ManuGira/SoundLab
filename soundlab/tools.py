from scipy.io import wavfile
import scipy.signal


def read_wav(filename, target_fs, target_win_size):
    """
    Read wav file and format it:
     - resample it to target_fs
     - crop it down to nearest multiple of target_win_size
    :param filename:
    :param target_fs: desired frequency sampling
    :param target_win_size:
    :return:
    """
    fs, data_int16 = wavfile.read(filename)
    if len(data_int16.shape) > 1:
        data_int16 = data_int16[:, 0]
    data = data_int16.copy() / (2 ** 15 - 1)

    data = scipy.signal.resample_poly(data, up=target_fs, down=fs)

    n = len(data)
    n -= n % target_win_size
    data = data[:n]

    return data


def write_wav(filename, data, target_fs):
    data_int16 = (data * (2 ** 15 - 1)).astype(np.int16)
    wavfile.write(filename, target_fs, data_int16)
