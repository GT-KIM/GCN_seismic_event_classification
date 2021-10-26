import numpy as np
import os

def data_extract(data, num_sample=3500):
    dataE = data[:,0]
    dataE = dataE - np.mean(dataE)
    dataE = dataE[:num_sample]
    dataN = data[:,1]
    dataN = dataN - np.mean(dataN)
    dataN = dataN[:num_sample]
    dataZ = data[:,2]
    dataZ = dataZ - np.mean(dataZ)
    dataZ = dataZ[:num_sample]
    data = np.concatenate((dataE[..., np.newaxis],
                           dataN[..., np.newaxis],
                           dataZ[..., np.newaxis]), axis=-1)
    return data

rootpath = "/sdc/EqData/CNNdata/"
save_rootpath = "/sdc/Eq2020_multisite_0925/CNNData/"
years = ['2016Data','2017Data', '2018Data', '2019Data']
for year in years :
    dataHH = np.load(rootpath + year + "_HH_3.npz")['data']
    dataHG = np.load(rootpath + year + "_HG_3.npz")['data']
    dataEL = np.load(rootpath + year + "_EL_3.npz")['data']

    savepath = save_rootpath + year + "/3/"
    if not os.path.isdir(savepath) :
        os.makedirs(savepath)

    data = np.concatenate((dataHH, dataHG, dataEL), axis=0)
    for i in range(10000) :
        idx = np.random.randint(0, 59999, 10)
        spec_data = list()
        for j in range(3) :
            curr_data = data_extract(data[idx[j],:,:])
            spec_data.append(curr_data)
        spec_data = np.array(spec_data)
        np.savez(savepath + str(i) + ".npz", data=spec_data, label=3)


