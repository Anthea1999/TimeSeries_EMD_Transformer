import torch

from sklearn.metrics import confusion_matrix
from torch import nn,optim 
from ecg_dataset import MyTestDataLoader
import numpy as np

from models.model.transformer import Transformer

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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

# Your existing code
batch_size = 10
test_loader = MyTestDataLoader(batch_size=batch_size).getDataLoader()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
sequence_len = 2500
max_len = 5000
n_head = 2
n_layer = 1
drop_prob = 0.1
d_model = 64
ffn_hidden = 128
feature = 1
model = Transformer(d_model=d_model, n_head=n_head, max_len=max_len, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, details=False, device=device).to(device=device)
model.load_state_dict(torch.load('1_best.pt')) 

# Run the model and get metrics
confusion_matrix = test_model_with_metrics(model, test_loader, device)

print("Confusion Matrix:")
print(confusion_matrix)

accuracy, precision, recall, f1_score = calculate_metrics(confusion_matrix)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)