import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import dgl
import torch
import numpy as np
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import obspy
import matplotlib.pyplot as plt
rootpath = "/sdc/Eq2020_multisite_0925/Case_study/case_study_multi/"
modelpath = "/sdc/Eq2020_multisite_0925/GCN_single/"

class MyDataset(Dataset) :
    def __init__(self, data_PHA2, data_YOCB, data_USN2) :
        self.data_PHA2 = data_PHA2
        self.data_YOCB = data_YOCB
        self.data_USN2 = data_USN2

    def __len__(self) :
        return 1

    def __getitem__(self, idx) :
        data_PHA2 = np.concatenate((self.data_PHA2[0].data[:,np.newaxis],
                               self.data_PHA2[1].data[:,np.newaxis],
                               self.data_PHA2[2].data[:,np.newaxis]), axis=1)

        data = data_PHA2[...,np.newaxis]

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

data_PHA2 = obspy.read(rootpath + "PHA2000")
data_YOCB = obspy.read(rootpath + "YOCB004")
data_USN2 = obspy.read(rootpath + "USN2002")

test_dataset = MyDataset(data_PHA2, data_YOCB, data_USN2)

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

class BidirectionalLSTM(nn.Module) :
    def __init__(self, nIn, nHidden, nOut) :
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T*b, h)

        output = self.embedding(t_rec)
        output = output.view(T, b, -1)

        return output

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn = collate)

model = Classifier()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
model.cuda(0)
model.train()
restore = True
if restore :
    model.load_state_dict(torch.load(modelpath + "bestGCNmodel.pth"))
    print("Load Finished!")
epoch_losses = []
print("test start")
result = np.zeros(3, dtype=np.int)
result_list = list()
for iter, (data, l) in enumerate(test_loader) :
    model.eval()
    idx = 0
    while idx < len(data) :
        bg = data[idx : idx + 1]
        label = l[idx : idx + 1]
        prediction = model(bg)
        _, indices = torch.max(prediction, dim=1)
        result[indices.cpu().numpy()] += 1
        result_list.append(indices.cpu().numpy())
        correct = torch.sum(indices == label)
        print(result)
        idx += 1
print(result)
test_all_1 = data_PHA2[2].data
test_all_1 = test_all_1 - np.mean(test_all_1)
result_list = np.array(result_list)
plt.subplot(211)
plt.title("[{} {} {}]".format(result[0], result[1], result[2]))
plt.plot(test_all_1)
plt.subplot(212)
plt.plot(result_list)
plt.savefig("/sdc/Eq2020_multisite_0925/case_study_single_event.jpg")
plt.show()