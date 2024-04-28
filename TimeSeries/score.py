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
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 622, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)  # Reshape the input to [batch_size, channels, sequence_length]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 622)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class VGGNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(VGGNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Adjust input shape from [batch_size, height, channels] to [batch_size, channels, height]
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Define a basic Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self.make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self.make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self.make_layer(256, 512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
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

# Your existing code
batch_size = 10
test_loader = MyTestDataLoader(batch_size=batch_size).getDataLoader()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_size = 1 # Define the input size based on your data
hidden_size = 64  # Change this value as needed
num_layers = 2     # Change this value as needed
output_size = 3 # Define the output size based on your classification
num_classes = 3
#input_channels = 1  # Modify this according to your data
input_channels = 2500 #ResNet

input = torch.randn(100,2500,1).to(device)


#model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
#model = GRUModel(input_size, hidden_size, num_layers, num_classes).to(device)
model = CNN1D().to(device)
#model = VGGNet(input_channels, num_classes).to(device)
#model = ResNet18(in_channels=input_channels, num_classes=num_classes).to(device)

model.load_state_dict(torch.load('CNN_best.pt')) 


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