
import torch

from torch.utils.tensorboard import SummaryWriter
import copy
from torch import nn,optim 
from torchinfo import summary 
from ecg_dataset import myDataLoader
import numpy as np 

import torch.nn.functional as F

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


# Assuming you have initialized your ResNet model, defined data loaders, and other necessary components...

def cross_entropy_loss(pred, target):

    criterion = nn.CrossEntropyLoss()
    #print('pred : '+ str(pred ) + ' target size: '+ str(target.size()) + 'target: '+ str(target )+   ' target2: '+ str(target))
    #print(  str(target.squeeze( -1)) )
    target= target.type(torch.cuda.LongTensor)
    lossClass= criterion(pred, target ) 

    return lossClass


def calc_loss_and_score(pred, target, metrics): 
    softmax = nn.Softmax(dim=1)

    pred =  pred.squeeze( -1)
    target= target.squeeze( -1)
    
    ce_loss = cross_entropy_loss(pred, target)
    #metrics['loss'] += ce_loss.data.cpu().numpy() * target.size(0)
    #metrics['loss'] += ce_loss.item()* target.size(0)
    metrics['loss'] .append( ce_loss.item() )
    pred = softmax(pred )
    
    #lossarr.append(ce_loss.item())
    #print('metrics : '+ str(ce_loss.item())  )
    #print('predicted max before = '+ str(pred))
    #pred = torch.sigmoid(pred)
    _,pred = torch.max(pred, dim=1)
    #print('predicted max = '+ str(pred ))
    #print('target = '+ str(target ))
    metrics['correct']  += torch.sum(pred ==target ).item()
    #print('correct sum =  '+ str(torch.sum(pred==target ).item()))
    metrics['total']  += target.size(0) 
    #print('target size  =  '+ str(target.size(0)) )

    return ce_loss
 
 
def print_metrics(main_metrics_train,main_metrics_val,metrics, phase):
   
    correct= metrics['correct']  
    total= metrics['total']  
    accuracy = 100*correct / total
    loss= metrics['loss'] 
    if(phase == 'train'):
        main_metrics_train['loss'].append( np.mean(loss)) 
        main_metrics_train['accuracy'].append( accuracy ) 
    else:
        main_metrics_val['loss'].append(np.mean(loss)) 
        main_metrics_val['accuracy'].append(accuracy ) 
    
    result = "phase: "+str(phase) \
    +  ' \nloss : {:4f}'.format(np.mean(loss))   +    ' accuracy : {:4f}'.format(accuracy)        +"\n"
    return result 

def train_model(dataloaders,model,optimizer, num_epochs=100): 
 
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    train_dict= dict()
    train_dict['loss']= list()
    train_dict['accuracy']= list() 
    val_dict= dict()
    val_dict['loss']= list()
    val_dict['accuracy']= list() 

    writer = SummaryWriter('./GRU-FCN')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10) 

        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = dict()
            metrics['loss']=list()
            metrics['correct']=0
            metrics['total']=0
 
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.int)
                # zero the parameter gradients
                optimizer.zero_grad()


                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    #print('outputs size: '+ str(outputs.size()) )
                    loss = calc_loss_and_score(outputs, labels, metrics)   
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                #print('epoch samples: '+ str(epoch_samples)) 
            

            if(phase == 'train'):
                writer.add_scalar('train_loss', np.mean(metrics['loss']), epoch)
                writer.add_scalar('train_accuracy', ((100 * metrics['correct']) / metrics['total']), epoch)

            else:
                writer.add_scalar('val_loss', np.mean(metrics['loss']), epoch)
                writer.add_scalar('val_accuracy', ((100 * metrics['correct']) / metrics['total']), epoch)


            print(print_metrics(main_metrics_train=train_dict, main_metrics_val=val_dict,metrics=metrics,phase=phase ))
            epoch_loss = np.mean(metrics['loss'])
        
            if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 'GRU-FCN_best.pt')

    writer.close()

    print('Best val loss: {:4f}'.format(best_loss))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


input_size = 1  # Modify this according to your data
hidden_size = 64
num_classes = 3    # Change this based on the number of classes in your dataset
model = GRU_FCN_Model(input_size, hidden_size, num_classes).to(device)

batch_size = 50
optimizer = optim.Adam(model.parameters(), lr=0.001)
dataloaders= myDataLoader(batch_size=batch_size).getDataLoader()


model_normal_ce = train_model(dataloaders=dataloaders,model=model,optimizer=optimizer, num_epochs=50)
torch.save(model.state_dict(), 'GRU-FCN.pt')