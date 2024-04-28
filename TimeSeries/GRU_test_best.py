
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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_size = 1 # Specify the input size
hidden_size = 64  # Specify the hidden size
num_layers = 2  # Specify the number of GRU layers
num_classes = 3  # Specify the number of output classes

# Initialize the model
model = GRUModel(input_size, hidden_size, num_layers, num_classes).to(device)

model.load_state_dict(torch.load('GRU_best.pt')) 

test_model(device=device, model=model, test_loader=test_loader)