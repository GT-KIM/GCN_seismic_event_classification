import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import dgl
import torch
import numpy as np
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import glob
import obspy
import matplotlib.pyplot as plt
rootpath = "/sdc/Eq2020_multisite_0925/Case_study_all/case_study_multi/"
modelpath = "/sdc/Eq2020_multisite_0925/GCN_single/"

class MyDataset(Dataset) :
    def __init__(self, data_000, data_001, data_002) :
        self.data_000 = data_000
        self.data_001 = data_001
        self.data_002 = data_002

    def __len__(self) :
        return 1

    def __getitem__(self, idx) :
        data_000 = np.concatenate((self.data_000[0].data[:,np.newaxis],
                               self.data_000[1].data[:,np.newaxis],
                               self.data_000[2].data[:,np.newaxis]), axis=1)

        data = data_000[...,np.newaxis]

        l = 2

        one_hot_train_y = np.zeros((1,3))
        one_hot_train_y[0,l] = 1.

        feature = np.transpose(data, (2, 1, 0))
        feature = feature[:3,:,:]
        time = 0
        samples = list()
        while time < feature.shape[2]-3000 :
            f = feature[:, :, time:time+3000]
            for i in range(f.shape[1]) :
                f[:,i,:] = f[:,i,:] - np.mean(f[:,i,:])

            sample = (f, l)
            samples.append(sample)
            time += 100
        return samples
from sklearn.metrics import confusion_matrix

def collate(samples) :
    samples = samples[0]
    data, labels = map(list, zip(*samples))
    return torch.tensor(data).float().cuda(), torch.tensor(labels).cuda()

def evaluate(model, g, labels) :
    model.eval()
    with torch.no_grad() :
        logits = model(g)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        gt = labels.cpu()
        pred = indices.cpu()
        CM = confusion_matrix(pred, gt,(0,1,2,3))

        return correct.item() * 1.0 / len(labels), CM

class FeatureExtraction(nn.Module) :
    def __init__(self) :
        super(FeatureExtraction, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv1_pool = nn.MaxPool1d(3, 2, 1)
        self.conv2 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv2_pool = nn.MaxPool1d(3, 2, 1)
        self.conv3 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv3_pool = nn.MaxPool1d(3, 2, 1)
        self.conv4 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv4_pool = nn.MaxPool1d(3, 2, 1)
        self.conv5 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv5_pool = nn.MaxPool1d(3, 2, 1)
        self.conv6 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv6_pool = nn.MaxPool1d(3, 2, 1)
        self.conv7 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv7_pool = nn.MaxPool1d(3, 2, 1)
        self.conv8 = nn.Conv1d(64, 64, 3, 1, 1)
        self.conv8_pool = nn.MaxPool1d(3, 2, 1)

    def forward(self, feature) :
        feature = self.conv1_pool(F.relu(self.bn1(self.conv1(feature))))
        feature = self.conv2_pool(F.relu((self.conv2(feature))))
        feature = self.conv3_pool(F.relu((self.conv3(feature))))
        feature = self.conv4_pool(F.relu((self.conv4(feature))))
        feature = self.conv5_pool(F.relu((self.conv5(feature))))
        feature = self.conv6_pool(F.relu((self.conv6(feature))))
        feature = self.conv7_pool(F.relu((self.conv7(feature))))
        outputs = self.conv8_pool(F.relu((self.conv8(feature))))
        outputs = torch.flatten(outputs, start_dim=1)
        return outputs

class Classifier(nn.Module) :
    def __init__(self) :
        super(Classifier, self).__init__()
        self.feature_extract = FeatureExtraction()
        self.dense1 = nn.Linear(768, 256)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 3)

    def forward(self, x) :
        x = self.feature_extract(x[:,0,:,:])
        x = self.dense1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x

model = Classifier()
model.cuda(0)
restore = True
if restore :
    model.load_state_dict(torch.load(modelpath + "bestGCNmodel.pth"))
    print("Load Finished!")

event_list = os.listdir(rootpath)
avg_acc = 0.
avg_false = 0
event_idx = 0
for event in event_list :
    print(event)
    eventpath = rootpath + event + "/"
    file_list = os.listdir(eventpath)
    if len(file_list) == 3 :
        try :
            data_000 = obspy.read(glob.glob(eventpath + "*000")[0])
            data_001 = obspy.read(glob.glob(eventpath + "*001")[0])
            data_002 = obspy.read(glob.glob(eventpath + "*002")[0])
            if len(data_000[0].data) == len(data_000[1].data) and len(data_000[0].data) == len(data_000[2].data) and\
                len(data_000[1].data) == len(data_000[2].data) :
                if len(data_001[0].data) == len(data_001[1].data) and len(data_001[0].data) == len(data_001[2].data) and \
                        len(data_001[1].data) == len(data_001[2].data):
                    if len(data_002[0].data) == len(data_002[1].data) and len(data_002[0].data) == len(data_002[2].data) and \
                            len(data_002[1].data) == len(data_002[2].data):

                        test_dataset = MyDataset(data_000, data_001, data_002)
                        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn = collate)

                        epoch_losses = []
                        print("test start")
                        result = np.zeros(3, dtype=np.int)
                        result_list = list()
                        for iter, (data, l) in enumerate(test_loader) :
                            model.eval()
                            idx = 0
                            while idx < len(data) :
                                bg = data[idx: idx + 1]
                                label = l[idx: idx + 1]
                                prediction = model(bg)
                                _, indices = torch.max(prediction, dim=1)
                                result[indices.cpu().numpy()] += 1
                                result_list.append(indices.cpu().numpy())
                                correct = torch.sum(indices == label)
                                #print(result)
                                idx += 1
                            accuracy = result[2] / len(data) * 100
                        if len(data) > 0 :
                            print(result)
                            test_all_1 = data_000[2].data
                            test_all_1 = test_all_1 - np.mean(test_all_1)
                            test_all_2 = data_001[2].data
                            test_all_2 = test_all_2 - np.mean(test_all_2)
                            test_all_3 = data_002[2].data
                            test_all_3 = test_all_3 - np.mean(test_all_3)
                            result_list = np.array(result_list)
                            fig, axes = plt.subplots(2, 1, sharex=False, sharey=False, figsize=(18, 18))
                            fig.suptitle("[{} {} {}]".format(result[0], result[1], result[2]))
                            x = np.arange(len(test_all_1)) / len(test_all_1) * 12
                            axes[0].plot(x, test_all_1)
                            x = np.arange(len(result_list)) / len(result_list) * 12
                            axes[1].plot(x, result_list)
                            plt.yticks(np.arange(0, 3), np.arange(0, 3))
                            if not os.path.isdir("/sdc/Eq2020_multisite_0925/case_study_new/noise_single/"+event+"/") :
                                os.makedirs("/sdc/Eq2020_multisite_0925/case_study_new/noise_single/"+event+"/")
                            plt.savefig("/sdc/Eq2020_multisite_0925/case_study_new/noise_single/"+event+"/case_study_noise.jpg")
                            plt.clf()
                            #plt.show()
                            event_idx += 1
                            avg_acc += accuracy
                            avg_false += result[0]
                            avg_false += result[1]
        except :
            print("file error")
avg_acc /= event_idx
avg_false /= event_idx
print("average accuracy : %f"%(avg_acc))
print("average the number of false alarm : %f"%(avg_false))