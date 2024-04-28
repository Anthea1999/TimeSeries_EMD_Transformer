
import torch

from torch import nn,optim 
from ecg_dataset import MyTestDataLoader
import numpy as np

from models.model.transformer import Transformer

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
batch_size = 10

test_loader = MyTestDataLoader(batch_size=batch_size).getDataLoader()

class GRU_FCN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GRU_FCN_Model, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.conv1 = nn.Conv1d(hidden_size, 128, kernel_size=5)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=5)
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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_size = 1  # Modify this according to your data
hidden_size = 64
num_classes = 3    # Change this based on the number of classes in your dataset
model = GRU_FCN_Model(input_size, hidden_size, num_classes).to(device)

model.load_state_dict(torch.load('GRU-FCN.pt')) 

test_model(device=device, model=model, test_loader=test_loader)