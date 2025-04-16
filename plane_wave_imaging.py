import os
import cv2
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets.PlaneWaveData import WUSData
from beamforming.DAS import DAS_PW

P = WUSData('configs\linear_array_7.5M.json')

rfdata = []
for j in range(len(P.angles)):
    df = pd.read_csv(f'rfdata/rfdata_1_{j+1}.csv', sep=',', header=None)
    data = df.values
    data = (data - 512) / 512
    data = data.T
    rfdata.append(data)

start_time = time.time()

rfdata = np.array(rfdata)
P.load_data(rfdata)

das = DAS_PW(P)

bimg = das.forward(P.rfdata)
print(time.time() - start_time)

cv2.imwrite('usimage.png', bimg)