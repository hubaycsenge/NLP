print('Starting training_script.py')
from tools.generator import NLPDataset
from tools.custom_models import RNN,LSTM
from tools.processor_functions import train_with_translate
from tools.processor_functions import evaluate_with_translate
from tools.optimizers import SimulatedAnnealing,GaussianSampler

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader

import gensim.downloader

print("Imports done")

train_ds = NLPDataset('data/train_1.pt')
val_ds = NLPDataset('data/val.pt')
print('Datasets computed')

train_loader = DataLoader(train_ds)
val_loader = DataLoader(val_ds)
print('Dataloaders ready')

gs_w2v_pretrained = gensim.downloader.load('word2vec-google-news-300')
print('W2v model loaded')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

n_epochs = 100
feature_num = 300

'''model = RNN(input_size = feature_num,hidden_size = feature_num).to(device) 
n_hidden = 1'''

model = nn.LSTM(input_size = 300,hidden_size = 300,batch_first = True).to(device)
n_hidden = 2
'''gs = GaussianSampler(mu=0, sigma=1, cuda=torch.cuda.is_available())
optimizer = SimulatedAnnealing(model.parameters(),sampler = gs)#optim.AdamW(model.parameters(), lr=lr)''' 

'''model = nn.Transformer(batch_first = True,nhead=10,d_model=feature_num).to(device)
n_hidden = 0'''
lr = 10
optimizer = optim.SGD(model.parameters(),lr=lr) #optim.AdamW(model.parameters(), lr=lr)
criterion = nn.CosineEmbeddingLoss(margin = 0.5) #nn.MSELoss(reduction='sum')
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {num_params:,} trainable parameters')

# RNN OR LSTM
hidden, train_loss = train_with_translate(model,gs_w2v_pretrained,optimizer,criterion,train_loader,-1 ,device,hidden_exists = True,n_hiddens= n_hidden,verbose = True)
lossdict = {"train_0" : train_loss}
with open('LSTM_05_24_losses.json', 'w') as outfile:
    json.dump(lossdict,outfile)
for epoch in range(n_epochs):
    hidden,train_loss = train_with_translate(model,gs_w2v_pretrained,optimizer,criterion,train_loader,epoch,device,hidden_exists = True,n_hiddens= n_hidden,hidden = hidden,verbose = False)
    val_loss = evaluate_with_translate(model,gs_w2v_pretrained,criterion,val_loader,epoch,device,hidden_exists = True,n_hiddens=n_hidden,hidden = hidden)
    lossdict[f"train_{epoch+1}"] = train_loss
    lossdict[f"val_{epoch+1}"] = val_loss
    with open('LSTM_05_31_losses.json', 'w') as outfile:
        json.dump(lossdict,outfile)
    if n_hidden == 1:
            torch.save(model,f'RNN_model_epoch={epoch}_={train_loss}.pt')
    else:
        torch.save(model,f'LSTM_model_epoch={epoch}_loss={train_loss}.pt')
    ''' if epoch % 10 == 0:
        #evaluate(model,optimizer,criterion,val_loader,epoch,device,hidden_exists = hiddenexists,n_hiddens=2,hidden = hidden) 
        if n_hidden == 1:
            torch.save(model,f'RNN_model_epoch={epoch}_={train_loss}.pt')
        else:
            torch.save(model,f'LSTM_model_epoch={epoch}_loss={train_loss}.pt')
        pass'''
    
'''# Transformer
train_loss = train_with_translate(model,gs_w2v_pretrained,optimizer,criterion,train_loader,-1 ,device,hidden_exists = False,verbose = False)
lossdict = {"train_0" : train_loss}
with open('Transformer_05_29_losses.json', 'w') as outfile:
    json.dump(lossdict,outfile)
for epoch in range(n_epochs):
    train_loss = train_with_translate(model,gs_w2v_pretrained,optimizer,criterion,train_loader,epoch,device,hidden_exists = False,verbose = False)
    val_loss = evaluate_with_translate(model,gs_w2v_pretrained,criterion,val_loader,epoch,device,hidden_exists = False)
    lossdict[f"train_{epoch+1}"] = train_loss
    lossdict[f"val_{epoch+1}"]  = val_loss
    with open('Transformer_05_25_losses.json', 'w') as outfile:
        json.dump(lossdict,outfile)
    #torch.save(model,f'Transformer_model_epoch={epoch}_loss={train_loss}.pt')
    if epoch % 10 == 0:
        torch.save(model,f'Transformer_model_epoch={epoch}_loss={train_loss}.pt')'''