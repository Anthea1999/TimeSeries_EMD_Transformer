
import torch
from thop import profile

from torchinfo import summary 
from models.model.transformer import Transformer
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
sequence_len=2500 # sequence length of time series
max_len=5000 # max time series sequence length 
n_head = 2 # number of attention head
n_layer = 1 # number of encoder layer
drop_prob = 0.1
d_model = 128 # number of dimension ( for positional embedding)
ffn_hidden = 512 # size of hidden layer before classification 
feature = 1 # for univariate time series (1d), it must be adjusted for 1. 

model =  Transformer(  d_model=d_model, details=True, n_head=n_head, max_len=max_len, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, device=device)

batch_size = 50
input = torch.randn(batch_size,sequence_len,feature).to(device)

summary(model, input_size=(batch_size,sequence_len,feature) , device=device)
flops, params = profile(model, inputs=(input,))


print(model)
print("flops:",flops)
print("params:",params)