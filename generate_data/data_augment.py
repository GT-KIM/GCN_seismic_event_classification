import numpy as np
import os
import matplotlib.pyplot as plt

rootpath = '/sdd1/Eq2020_multisite_0925/'
datapath = rootpath + 'CNNData/'
savepath = rootpath + 'CNNData_augment/'

years = ['2016Data','2017Data','2018Data','2019Data']
events = ['0', '1', '2']

for year in years :
    yearpath = datapath + year + "/"
    for event in events :
        eventpath = yearpath + event + "/"
        idx = 0
        eventlist = os.listdir(eventpath)
        for name in eventlist :
            data = eventpath + name
            d = np.load(data)['data']
            label = np.load(data)['label']
            if not os.path.isdir(savepath + year + '/' + event + '/') :
                os.makedirs(savepath + year + '/' + event + '/')
            for i in range(6) :
                augmented = d[:, i*100:i*100 + 3000, :] # [multi-station, length, channel]
                np.savez(savepath + year + '/' + event + '/' + str(idx) + '.npz',
                         data=augmented, label=label)
                idx += 1