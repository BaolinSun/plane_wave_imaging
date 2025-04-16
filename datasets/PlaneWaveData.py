# File:       PlaneWaveData.py
# Author:     SunBaolin
# Created on: 2025-04-14

import os
import json
import numpy as np

from scipy.fft import fft, ifft
from scipy.signal import hilbert

class PlaneWaveData:
    """ A template class that contains the plane wave data.

    PlaneWaveData is a container or dataclass that holds all of the information
    describing a plane wave acquisition. Users should create a subclass that reimplements
    __init__() according to how their data is stored.

    The required information is:
    idata       In-phase (real) data with shape (nangles, nchans, nsamps)
    qdata       Quadrature (imag) data with shape (nangles, nchans, nsamps)
    angles      List of angles [radians]
    ele_pos     Element positions with shape (N,3) [m]
    fc          Center frequency [Hz]
    fs          Sampling frequency [Hz]
    fdemod      Demodulation frequency, if data is demodulated [Hz]
    c           Speed of sound [m/s]
    time_zero   List of time zeroes for each acquisition [s]
    
    Correct implementation can be checked by using the validate() method. See the
    PICMUSData class for a fully implemented example.
    """

    def __init__(self):
        """ Users must re-implement this function to load their own data. """
        # Do not actually use PlaneWaveData.__init__() as is.
        raise NotImplementedError

        # We provide the following as a visual example for a __init__() method.
        nangles, nchans, nsamps = 2, 3, 4
        # Initialize the parameters that *must* be populated by child classes.
        # self.idata = np.zeros((nangles, nchans, nsamps), dtype="float32")
        # self.qdata = np.zeros((nangles, nchans, nsamps), dtype="float32")
        self.rfdata = np.zeros((nangles, nchans, nsamps), dtype="float32")
        self.angles = np.zeros((nangles,), dtype="float32")
        self.ele_pos = np.zeros((nchans, 3), dtype="float32")
        self.fc = 5e6
        self.fs = 20e6
        self.fdemod = 0
        self.c = 1540
        self.time_zero = np.zeros((nangles,), dtype="float32")

    def validate(self):
        """ Check to make sure that all information is loaded and valid. """
        # Check size of idata, qdata, angles, ele_pos
        assert self.rfdata.ndim == 3
        nangles, nchans, nsamps = self.rfdata.shape
        # assert self.angles.ndim == nangles
        # assert self.ele_pos.ndim == 2 and self.ele_pos.shape == (nchans, 3)
        # Check frequencies (expecting more than 0.1 MHz)
        assert self.fc > 1e5
        assert self.fs > 1e5
        assert self.fdemod > 1e5 or self.fdemod == 0
        # Check speed of sound (should be between 1000-2000 for medical imaging)
        assert 1000 <= self.c <= 2000
        # Check that a separate time zero is provided for each transmit
        # assert self.time_zero.ndim == 1 and self.time_zero.size == nangles


class WUSData(PlaneWaveData):
    def __init__(self, config_file):

        with open(config_file, "r") as file:
            probe_params = json.load(file)

        self.element_num = probe_params["element_num"]
        self.fc = probe_params["fc"]
        self.fs = probe_params["fs"]
        self.c = probe_params["c"]
        self.pitch = probe_params["pitch"]
        self.angles = probe_params['angles']
        self.sample_num = probe_params["sample_num"]
        self.drange = probe_params["drange"]
        self.fdemod = probe_params["fdemod"]
        self.scan_width = probe_params["scan_width"]
        self.scan_depth = probe_params["scan_depth"]
        self.scan_start_depth = probe_params["scan_start_depth"]
        self.x_axis_pixel = probe_params["x_axis_pixel"]
        self.z_axis_pixel = probe_params["z_axis_pixel"]

        self.ele_pos = np.zeros((self.element_num, 3), dtype="float32")
        self.ele_pos[:, 0] = np.arange(self.element_num) * self.pitch
        self.ele_pos[:, 0] -= np.mean(self.ele_pos[:, 0])

        self.tstart = np.ones((len(self.angles), self.element_num)) * 4.6914110470588584e-06

        self.rfdata = np.zeros((len(self.angles), self.element_num, self.sample_num), dtype="float32")
        self.validate()

    
    def load_data(self, data):

        for n in range(len(self.angles)):
            for i in range(self.element_num):
                data[n, i, :] = self.bandpass_filter_rf_data(data[n, i, :], data.shape[2], self.fs, 1.0e6, 5.0e6)

        rfdata = hilbert(data, axis=-1)
        self.rfdata = rfdata


    # 带通滤波器
    def bandpass_filter_rf_data(self, x, length, sampling_frequency, low_cutoff, high_cutoff):
        size = int(length)

        fft_result = fft(x)

        df = sampling_frequency / size  # 计算频率分辨率

        # 创建频率数组
        frequencies = np.arange(size) * df

        # 应用带通滤波器
        mask = (frequencies >= low_cutoff) & (frequencies <= high_cutoff)
        fft_result[~mask] = 0.0  # 将不在范围内的频率分量置零

        res = ifft(fft_result)

        return np.real(res)