import os
import cupy as cp
import numpy as np

from scipy.interpolate import interp1d


class DAS_PW():

    def __init__(self, P):

        self.fc = P.fc
        self.fs = P.fs
        self.c = P.c
        self.angles = np.radians(P.angles)
        self.ele_pos = P.ele_pos
        self.element_num = P.element_num
        self.sample_num = P.sample_num
        self.drange = P.drange
        self.x_axis_pixel = P.x_axis_pixel
        self.z_axis_pixel = P.z_axis_pixel
        self.tstart = P.tstart

        self.scan_x_axis = np.linspace(-P.scan_width/2, P.scan_width/2, P.x_axis_pixel, endpoint=True)
        self.scan_z_axis = np.linspace(P.scan_start_depth, P.scan_start_depth + P.scan_depth, P.z_axis_pixel, endpoint=True)
        self.scan_x_grid, self.scan_z_grid = np.meshgrid(self.scan_x_axis, self.scan_z_axis, indexing='ij')

        self.scan_x_grid = self.scan_x_grid.ravel()
        self.scan_z_grid = self.scan_z_grid.ravel()

        nangles = len(self.angles)
        nelems = self.element_num
        npixels = self.scan_x_grid.shape[0]

        self.txdel = np.zeros((nangles, npixels))
        self.rxdel = np.zeros((nelems, npixels))
        self.delays = np.zeros((nangles, nelems, npixels))
        for i, angle in enumerate(self.angles):
            self.txdel[i] = self.scan_x_grid * np.sin(angle) + self.scan_z_grid * np.cos(angle)
        
        for i in range(self.element_num):
            ele_x = self.ele_pos[i, 0]
            ele_z = self.ele_pos[i, 2]
            self.rxdel[i] = np.sqrt((self.scan_x_grid-ele_x)**2 + (self.scan_z_grid-ele_z)**2)
        
        for i, angle in enumerate(self.angles):
            for j in range(self.element_num):
                self.delays[i, j] = ((self.txdel[i] + self.rxdel[j]) / self.c - self.tstart[i][j]) * self.fs


    def forward(self, x):
        
        bf = np.zeros((self.x_axis_pixel, self.z_axis_pixel), dtype=x.dtype)

        for i, angle in enumerate(self.angles):
            rfdata = x[i]
            complex_enveloped = np.zeros((self.x_axis_pixel, self.z_axis_pixel), dtype=rfdata.dtype)
            for j in range(self.element_num):
                delays = self.delays[i, j]
                # interpolator = interp1d(np.arange(self.sample_num), rfdata[j], kind='linear', bounds_error=False,fill_value=0.0)
                # foc = interpolator(delays).reshape(self.x_axis_pixel, self.z_axis_pixel)
                foc = np.interp(delays, np.arange(self.sample_num), rfdata[j]).reshape(self.x_axis_pixel, self.z_axis_pixel)
                complex_enveloped += foc
            
            bf += complex_enveloped

        bf = np.abs(bf)

        bimg = self.make_image(bf)

        return bimg


    def make_image(self, bf):

        max_val = np.amax(bf)
        bimg = 20 * np.log10(bf/max_val)

        min_dB, max_dB = -self.drange, 0
        bimg = np.clip(bimg, min_dB, max_dB)

        bimg = 255 * (bimg - min_dB) / (max_dB - min_dB)
        bimg = np.clip(bimg, 0, 255).astype(np.uint8)

        return bimg.T


        


        


    # def delay_plane()
