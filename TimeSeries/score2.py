import torch
import torch.nn.functional as F
from torchinfo import summary

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch import nn,optim 
from ecg_dataset import MyTestDataLoader
import numpy as np

from models.model.transformer import Transformer

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from thop import profile

def cross_entropy_loss(pred, target): 
    criterion = nn.CrossEntropyLoss() 
    target= target.type(torch.cuda.LongTensor)
    lossClass= criterion(pred, target)  
    return lossClass 

def calc_loss_and_score(pred, target, metrics): 
    softmax = nn.Softmax(dim=1)

    pred =  pred.squeeze(-1)
    target= target.squeeze(-1)
    
    ce_loss = cross_entropy_loss(pred, target) 

    metrics['loss'] .append( ce_loss.item() )
    pred = softmax(pred ) 
    _,pred = torch.max(pred, dim=1) 
    correct = torch.sum(pred ==target ).item() 
    metrics['correct']  += correct
    total = target.size(0)   
    metrics['total']  += total
    print('loss : ' +str(ce_loss.item() ) + ' correct: ' + str(((100 * correct )/total))  + ' target: ' + str(target.data.cpu().numpy()) + ' prediction: ' + str(pred.data.cpu().numpy()))
    return ce_loss

def print_average(metrics):  

    loss= metrics['loss'] 

    print('average loss : ' +str(np.mean(loss))  + ' average correct: ' + str(((100 * metrics['correct']  )/ metrics['total']) ))
 

def test_model(model,test_loader,device):
    model.eval() 
    metrics = dict()
    metrics['loss']=list()
    metrics['correct']=0
    metrics['total']=0
    for inputs, labels in test_loader:
        with torch.no_grad():
            
            inputs = inputs.to(device=device, dtype=torch.float )
            labels = labels.to(device=device, dtype=torch.int) 
            pred = model(inputs) 
            
            calc_loss_and_score(pred, labels, metrics) 
    print_average(metrics)

def test_model_with_metrics(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for inputs, labels in test_loader:
        with torch.no_grad():
            inputs = inputs.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.int)
            pred = model(inputs)
            pred = torch.argmax(pred, dim=1).cpu().numpy()
            all_preds.extend(pred)
            all_labels.extend(labels.cpu().numpy())
   
    # Confusion matrix
    confusion = confusion_matrix(all_labels, all_preds) 
    
    return confusion

def calculate_metrics(confusion_matrix):
    TP = confusion_matrix[0][0]
    TN = confusion_matrix[1][1] + confusion_matrix[1][2] + confusion_matrix[2][1] + confusion_matrix[2][2]
    FP = confusion_matrix[1][0] + confusion_matrix[2][0]
    FN = confusion_matrix[0][1] + confusion_matrix[0][2] + confusion_matrix[1][1] + confusion_matrix[1][2] + confusion_matrix[2][1] + confusion_matrix[2][2]

    accuracy = (TP + TN) / sum(sum(row) for row in confusion_matrix)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score

class GRU_FCN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GRU_FCN_Model, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.conv1 = nn.Conv1d(hidden_size, 128, kernel_size=7)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=7)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=7)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(3)
        self.globalavgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = out.permute(0, 2, 1)
        out = self.relu(self.conv1(out))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.maxpool(out)
        out = self.globalavgpool(out)
        out = out.view(-1, 128)
        out = self.fc(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Define the ResNet model
class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear(x)
        return self.sigmoid(out)


# Your existing code
batch_size = 10
test_loader = MyTestDataLoader(batch_size=batch_size).getDataLoader()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


input_size = 1  # Modify this according to your data
hidden_size = 32
num_classes = 3    # Change this based on the number of classes in your dataset
input_channels = 2500

input = torch.randn(100,2500,1).to(device)

#model = GRU_FCN_Model(input_size, hidden_size, num_classes).to(device)
model = ResNet(ResidualBlock, [3, 4, 6, 3], in_channels=input_channels, num_classes=num_classes).to(device)
#model = LogisticRegression(input_size, num_classes).to(device)

model.load_state_dict(torch.load('ResNet50-1_best.pt'))  


# Run the model and get metrics
confusion_matrix = test_model_with_metrics(model, test_loader, device)

print("Confusion Matrix:")
print(confusion_matrix)

accuracy, precision, recall, f1_score = calculate_metrics(confusion_matrix)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
summary(model, input_size=(100,2500,1) , device=device)
flops, params = profile(model, inputs=(input,))
print("flops:",flops)
print("params:",params)