# -*- coding: utf-8 -*-
from obspy import read, Trace, Stream, UTCDateTime
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import detrend, spectrogram
from scipy.linalg import hankel
from numpy.linalg import svd
from operator import itemgetter
from itertools import permutations, combinations
import random

def data_extract(data_in_list, i,sample):
    dataE = data_in_list[i]['tr_E']
    dataE = dataE - np.mean(dataE)
    dataE = dataE[sample:sample+3500]
    dataN = data_in_list[i]['tr_N']
    dataN = dataN - np.mean(dataN)
    dataN = dataN[sample:sample+3500]
    dataZ = data_in_list[i]['tr_Z']
    dataZ = dataZ - np.mean(dataZ)
    dataZ = dataZ[sample:sample+3500]
    data = np.concatenate((dataE[..., np.newaxis],
                           dataN[..., np.newaxis],
                           dataZ[..., np.newaxis]), axis=-1)
    return data


FLAGS = ["event"]
sample_rate = 100
labels = ['0', '1', '2']
years = ['2016Data','2017Data', '2018Data', '2018Data2', '2019Data']
rootpath = "/sdd1/Eq2020_multisite_0925/"
rootsavepath = "/sdd1/Eq2020_multisite_0925/CNNData/"
if not os.path.isdir(rootsavepath) :
    os.makedirs(rootsavepath)

count_HH = 0
count_HG = 0
count_EL = 0
x_2017 = list()
y_2017 = list()
x_2018 = list()
y_2018 = list()
method = 0
for year in years :
    yearpath = rootpath + year + '/'
    if not year == "2018Data2" :
        win_idx = 0
    for label in labels :
        labelpath = yearpath + label + '/'
        for FLAG in FLAGS :
            eventpath = labelpath + FLAG + '/'
            events = os.listdir(eventpath)
            for event in events :
                datapath = eventpath + event + '/'
                filenames = os.listdir(datapath)
                data_in_list = list()
                for filename in filenames :
                    filepath_E = os.path.join(datapath, filename)
                    if not os.path.isfile(filepath_E) :
                        continue
                    if os.path.getsize(filepath_E) <= 1 :
                        continue
                    d = dict(np.load(filepath_E))
                    data_in_list.append(d)
                data_in_list = sorted(data_in_list, key=itemgetter('station_TIME'))

                if method == 0 :
                    for i in range(len(data_in_list)-2) :
                        t1 = UTCDateTime(str(data_in_list[i]['station_DATE']) + 'T' + str(data_in_list[i]['station_TIME']))
                        t2 = UTCDateTime(str(data_in_list[i]['event_DATE']) + 'T' + str(data_in_list[i]['event_TIME']))
                        sample = int((t1 - t2)*100)
                        if sample > 3500 :
                            continue
                        #Extract Seed
                        data1 = data_extract(data_in_list, i, sample)
                        data2 = data_extract(data_in_list, i+1, sample)
                        data3 = data_extract(data_in_list, i+2, sample)

                        if label == '0' or label == '1' :
                            l = 0
                        elif label == '2' :
                            l = 1
                        else :
                            l = 99
                        if not l == 99:
                            savepath = rootsavepath + year + "/" + str(l) + "/"
                            if year == '2018Data2':
                                savepath = rootsavepath + "2018Data/" + str(l) + "/"
                            if not os.path.isdir(savepath):
                                os.makedirs(savepath)
                        data = np.concatenate((data1[..., np.newaxis],
                                               data2[..., np.newaxis],
                                               data3[..., np.newaxis]), axis=-1)
                        data = np.transpose(data, (2, 0, 1))
                        if not l == 99 :
                            np.savez(savepath + str(win_idx)+ ".npz" , data=data, label=l)
                            win_idx += 1
                            if win_idx % 100 == 0 :
                                print(win_idx)
print("============ Done ============")
