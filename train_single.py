import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import dgl
import torch
import numpy as np
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

rootpath = "/sdd1/Eq2020_multisite_0925/CNNData_augment/"
modelpath = "/sdd1/Eq2020_multisite_0925/GCN_single/"
if not os.path.isdir(modelpath) :
    os.makedirs(modelpath)
years = ['2016Data', '2017Data', '2018Data', '2019Data']
labels = [0, 1, 2]
class MyDataset(Dataset) :
    def __init__(self, data_list) :
        self.data_list = data_list

    def __len__(self) :
        return len(self.data_list)

    def __getitem__(self, idx) :
        d = np.load(self.data_list[idx], allow_pickle=False)
        data = d['data']
        label = d['label']
        if label == 0 :
            l = 0
        elif label == 1 :
            l = 1
        else :
            l = 2

        feature = np.transpose(data, (0, 2, 1))
        feature = feature[:1,:,:]

        return feature, l

train_list = list()
test_list = list()
for year in years :
    yearpath = rootpath + year + "/"
    for label in labels :
        labelpath = yearpath + str(label) + "/"
        datalist = os.listdir(labelpath)
        for dataname in datalist :
            datapath = labelpath + dataname
            if not year == "2018Data" :
                #d = np.load(datapath)
                #data = d['data']
                #label = d['label']
                train_list.append(datapath)
            else :
                test_list.append(datapath)

train_dataset = MyDataset(train_list)
test_dataset = MyDataset(test_list)

def collate(samples) :
    data, labels = zip(*samples)
    return torch.tensor(data).float().cuda(), torch.tensor(labels).cuda()

from sklearn.metrics import confusion_matrix

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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn = collate)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn = collate)

model = Classifier()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
model.cuda(0)
model.train()
restore = False
if restore :
    model.load_state_dict(torch.load(modelpath + "bestGCNmodel.pth"))
    print("Load Finished!")
epoch_losses = []
print("training start")
best_acc = 0.0001
for epoch in range(500) :
    epoch_loss = 0
    epoch_acc = 0.0
    #model.cuda(0)
    for iter, (bg, label) in enumerate(train_loader) :
        model.train()
        with torch.enable_grad() :
            prediction = model(bg)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        with torch.no_grad() :
            _, indices = torch.max(prediction, dim=1)
            correct = torch.sum(indices == label)
            acc = correct.item() * 1.0 / len(label)
            epoch_acc += acc
        #print(iter, loss.detach().item(), acc)
    epoch_loss /= (iter + 1)
    epoch_acc /= (iter + 1)
    test_acc = 0.0
    conMAT = np.zeros((4,4))
    for iter, (test_g, test_labels) in enumerate(test_loader) :
        acc, CM = evaluate(model, test_g, test_labels)
        test_acc += acc
        conMAT += CM
    test_acc /= (iter + 1)
    if test_acc >= best_acc :
        savepath = modelpath + "bestGCNmodel.pth"
        torch.save(model.state_dict(), savepath)
        best_acc = test_acc
        print("Saved test acc : %f"%best_acc)
    print('Epoch {}, loss {:.4f}, train acc : {} test acc : {}'.format(epoch, epoch_loss,epoch_acc, test_acc))
    print(conMAT)
    epoch_losses.append(epoch_loss)