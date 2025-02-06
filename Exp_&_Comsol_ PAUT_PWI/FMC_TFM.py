# @Deepak
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
from scipy.signal import hilbert

def sub_main(file_path):
    file = np.fromfile(file_path, dtype="<i2", sep='')
    transducer_num = int(file[55])
    data_points = int(file[63])
    return transducer_num, data_points, file

def get_matrix(file, transducer_num, data_points):
    file = file[64:]
    matrix = np.array([])
    for i in range(transducer_num):
        start = (data_points + 10) * i * transducer_num + i * 10
        end = start + (data_points + 10) * transducer_num
        asc = file[start:end]
        d = np.array([])
        d = np.append(d, asc)
        d = d.reshape(transducer_num, data_points + 10)
        matrix = np.append(matrix, d[:, :data_points])

    matrix = matrix.reshape(transducer_num, transducer_num, data_points)
    return matrix

def data_index(transducers, pixel_ind, velocity, frequency):
    dist = {}
    for i, tr in transducers.items():
        dist[i] = ((np.sqrt((tr[0] - pixel_ind[0]) ** 2 + (tr[1] - pixel_ind[1]) ** 2)) / velocity) * frequency
    return dist

def plot1(image, roi):
    color_list = [(0.0, 'white'), (0.33, 'blue'), (0.67, 'yellow'), (1.0, 'red')]
    cmap = colors.LinearSegmentedColormap.from_list('mycmap', color_list)
    plt.imshow(image, cmap=cmap, extent=[roi[0] * 1000, roi[1] * 1000, roi[3] * 1000, roi[2] * 1000])
    plt.colorbar()
    plt.show()

def fast_tfm(start_time, pitch, data_points, transducer_num, matrix, roi, pix_size, velocity, frequency):
    transducer_position = [((transducer - transducer_num / 2 + 0.5) * pitch, 0) for transducer in range(transducer_num)]
    transducers = dict(enumerate(transducer_position))
    width = int(round((roi[1] - roi[0]) / pix_size))
    height = int(round((roi[3] - roi[2]) / pix_size))
    x_pixel = np.linspace(roi[0], roi[1], width)
    y_pixel = np.linspace(roi[2], roi[3], height)
    pixel_ind = np.meshgrid(x_pixel, y_pixel)
    image_array = np.zeros((height, width))
    dp = data_index(transducers, pixel_ind, velocity, frequency)
    for t in range(transducer_num):
        for r in range(transducer_num):
            ai = dp[t] + dp[r]
            bi = ai.reshape(width * height).astype(int)
            amp = np.take(matrix[t][r], bi)
            amp = amp.reshape(height, width)
            image_array += amp
    bhil = hilbert_transform(image_array)
    return image_array

def hilbert_transform(signals):
    envelopes = np.zeros_like(signals)
    for i in range(signals.shape[1]):
        x = signals[:, i]
        x_hilbert = hilbert(x)
        x_env = np.abs(x_hilbert)
        envelopes[:, i] = x_env
    return envelopes

def main():
    start_time = time.time()
    file_path = r'D:\Py_Codes\Python FMC-TFM\Weld_exp_data\3 40db.capture_acq\data.bin'
    pix_size = 2/10000  # in meters
    pitch = 1/1000  # in meters
    velocity = 5700  # in m/s
    frequency = 100e6  # in Hz

    transducer_num, data_points, file = sub_main(file_path)
    roi = np.array([-40, 40, 0, 80]) / 1000  # in meters
    matrix = get_matrix(file, transducer_num, data_points)
    image = fast_tfm(start_time, pitch, data_points, transducer_num, matrix, roi, pix_size, velocity, frequency)
    hil_image = np.abs(hilbert_transform(image))
    plot1(hil_image, roi)

    
    hil_image = hil_image / np.max(hil_image)
    image2 = 20 * np.log10(hil_image + 1e-12)
    plot1(image2, roi)


if __name__ == "__main__":
    main()
