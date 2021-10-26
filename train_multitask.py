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
from sklearn.metrics import confusion_matrix

rootpath = "/sdc/Eq2020_multisite_0925/CNNData_augment/"
modelpath = "/sdc/Eq2020_multisite_0925/GCN_single/"
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
            label1 = 0
            label2 = 0
        elif label == 1 :
            label1 = 0
            label2 = 1
        else :
            label1 = 1
            label2 = 2

        feature = np.transpose(data, (0, 2, 1))
        feature = feature[:1,:,:]

        return feature, label1, label2

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
    data, label1, label2 = zip(*samples)
    return torch.tensor(data).float().cuda(), torch.tensor(label1).cuda(), torch.tensor(label2).cuda()


def evaluate(model, g, labels) :
    model.eval()
    with torch.no_grad() :
        _, logits = model(g)
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        gt = labels.cpu()
        pred = indices.cpu()
        CM = confusion_matrix(pred, gt,(0,1,2,3))

        return correct.item() * 1.0 / len(labels), CM

class AttentionModule(nn.Module) :
    def __init__(self, in_dim) :
        super(AttentionModule, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_dim, int(in_dim/4))
        self.fc2 = nn.Linear(int(in_dim/4), in_dim)

    def forward(self, x) :
        x_init = x
        x = self.global_pooling(x).squeeze(2)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x).unsqueeze(2)
        x = x_init * x

        return x

class FeatureAggregation(nn.Module) :
    def __init__(self, in_dim1, in_dim2, in_dim3) :
        super(FeatureAggregation, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool1d(1)

        dim1 = in_dim1 + in_dim2
        dim2 = in_dim2 + in_dim3
        self.fc11 = nn.Linear(dim1, int(dim1/4))
        self.fc12 = nn.Linear(int(dim1/4), dim1)
        self.fc21 = nn.Linear(dim2, int(dim2/4))
        self.fc22 = nn.Linear(int(dim2/4), dim2)

    def forward(self, x1, x2, x3) :
        x1 = self.global_pooling(x1).squeeze(2)
        x2 = self.global_pooling(x2).squeeze(2)
        x3 = self.global_pooling(x3).squeeze(2)

        path1 = torch.cat((x1, x2), dim=1)
        path1_init = path1
        path1 = self.fc11(path1)
        path1 = F.relu(path1, inplace=True)
        path1 = self.fc12(path1)
        path1 = torch.sigmoid(path1)
        path1 = path1_init * path1

        path2 = torch.cat((x2, x3), dim=1)
        path2_init = path2
        path2 = self.fc21(path2)
        path2 = F.relu(path2, inplace=True)
        path2 = self.fc22(path2)
        path2 = torch.sigmoid(path2)
        path2 = path2_init * path2

        out = torch.cat((path1, path2), dim=1)

        return out


class CNN_layer(nn.Module) :
    def __init__(self, in_dim, out_dim, is_bn = False) :
        super(CNN_layer, self).__init__()
        self.is_bn = is_bn
        self.conv1 = nn.Conv1d(in_dim, out_dim, 3, 1, 1)
        if self.is_bn :
            self.bn1 = nn.BatchNorm1d(out_dim)
        self.attention = AttentionModule(out_dim)
        self.max_pool = nn.MaxPool1d(2, 2, 0)

    def forward(self, x) :
        x = self.conv1(x)
        if self.is_bn :
            x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.attention(x)
        x = self.max_pool(x)
        return x

class Classifier(nn.Module) :
    def __init__(self) :
        super(Classifier, self).__init__()
        self.conv1 = CNN_layer(3, 32, is_bn=True)
        self.conv2 = CNN_layer(32, 32, is_bn=False)
        self.conv3 = CNN_layer(32, 32, is_bn=False)
        self.conv4 = CNN_layer(32, 32, is_bn=False)
        self.conv5 = CNN_layer(32, 32, is_bn=False)
        self.conv6 = CNN_layer(32, 32, is_bn=False)
        self.conv7 = CNN_layer(32, 32, is_bn=False)
        self.conv8 = CNN_layer(32, 32, is_bn=False)
        self.dropout = nn.Dropout(0.5)
        self.dense11 = nn.Linear(352, 128)
        self.dense12 = nn.Linear(128, 2)

        self.aggregation = FeatureAggregation(32, 32, 32)
        self.dense21 = nn.Linear(128, 128)
        self.dense22 = nn.Linear(128, 3)

    def forward(self, x) :
        x = x[:,0,:,:]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x1 = self.conv6(x)
        x2 = self.conv7(x1)
        x3 = self.conv8(x2)

        out1 = torch.flatten(x3, start_dim=1)
        out1 = F.relu(self.dense11(out1))
        out1 = self.dropout(out1)
        out1 = self.dense12(out1)

        out2 = self.aggregation(x1, x2, x3)
        out2 = F.relu(self.dense21(out2))
        out2 = self.dropout(out2)
        out2 = self.dense22(out2)

        return out1, out2

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
    for iter, (bg, label1, label2) in enumerate(train_loader) :
        model.train()
        with torch.enable_grad() :
            prediction1, prediction2 = model(bg)
            loss1 = loss_func(prediction1, label1)
            loss2 = loss_func(prediction2, label2)
            loss = loss1 +loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
        with torch.no_grad() :
            _, indices = torch.max(prediction2, dim=1)
            correct = torch.sum(indices == label2)
            acc = correct.item() * 1.0 / len(label2)
            epoch_acc += acc
        #print(iter, loss.detach().item(), acc)
    epoch_loss /= (iter + 1)
    epoch_acc /= (iter + 1)
    test_acc = 0.0
    conMAT = np.zeros((4,4))
    for iter, (test_g, _, test_labels) in enumerate(test_loader) :
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