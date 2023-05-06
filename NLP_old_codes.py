import numpy as np
import os
import time

text = open('shakespeare.txt', 'rb').read().decode(encoding='utf-8')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
 
# load ascii text and covert to lowercase
filename = "NLP/shakespeare.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
 
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
 
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
 
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
 
# reshape X to be [samples, time steps, features]
X = torch.tensor(dataX, dtype=torch.float32).reshape(n_patterns, seq_length, 1)
X = X / float(n_vocab)
y = torch.tensor(dataY)
 
class CharModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=256, num_layers=2, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(256, n_vocab)
    def forward(self, x):
        x, _ = self.lstm(x)
        # take only the last output
        x = x[:, -1, :]
        # produce output
        x = self.linear(self.dropout(x))
        return x
 
n_epochs = 40
batch_size = 128
model = CharModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
 
optimizer = optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss(reduction="sum")
loader = data.DataLoader(data.TensorDataset(X, y), shuffle=True, batch_size=batch_size)
 
best_model = None
best_loss = np.inf
for epoch in range(n_epochs):
    model.train()
    for X_batch, y_batch in loader:
        y_pred = model(X_batch.to(device))
        loss = loss_fn(y_pred, y_batch.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    model.eval()
    loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            y_pred = model(X_batch.to(device))
            loss += loss_fn(y_pred, y_batch.to(device))
        if loss < best_loss:
            best_loss = loss
            best_model = model.state_dict()
        print("Epoch %d: Cross-entropy: %.4f" % (epoch, loss))
 
torch.save([best_model, char_to_int], "single-char.pth")

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
n_vocab = len(char_to_int)
int_to_char = dict((i, c) for c, i in char_to_int.items())
model.load_state_dict(best_model)
 
# randomly generate a prompt
filename = "NLP/shakespeare.txt"
seq_length = 100
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()
start = np.random.randint(0, len(raw_text)-seq_length)
prompt = raw_text[start:start+seq_length]
pattern = [char_to_int[c] for c in prompt]
 
model.eval()
print('Prompt: "%s"' % prompt)
with torch.no_grad():
    for i in range(1000):
        # format input array of int into PyTorch tensor
        x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
        x = torch.tensor(x, dtype=torch.float32)
        # generate logits as output from the model
        prediction = model(x.to(device))
        # convert logits into one character
        index = int(prediction.argmax())
        result = int_to_char[index]
        print(result, end="")
        # append the new character into the prompt for the next iteration
        pattern.append(index)
        pattern = pattern[1:]
print()
print("Done.")

import re
import gc
import html
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import torchtext
from datasets import load_dataset
from tqdm import tqdm
from torchsummary import summary
from gensim.test.utils import common_texts
import gensim.downloader
from gensim.models import Word2Vec
from transformers import AutoTokenizer, Data2VecTextModel


!cp train_2.pt'/content/gdrive/My Drive'
!ls -lt '/content/gdrive/My Drive'


# dataloader functions https://towardsdatascience.com/language-modeling-with-lstms-in-pytorch-381a26badcbf
def preprocess_data(train_path,val_path,test_path,tokenizer):
    raw_dataset = load_dataset("text", data_files={"train": train_path,'val':val_path, "test": test_path}, sample_by="paragraph")
    tokenize_data = lambda example, tokenizer: {'tokens': tokenizer(example['text'])} 
    tokenized_dataset = raw_dataset.map(tokenize_data, remove_columns=['text'],fn_kwargs={'tokenizer': tokenizer}) 
    vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset['train']['tokens'],min_freq=3) 
    vocab.insert_token('<unk>', 0)           
    vocab.insert_token('<eos>', 1)            
    vocab.set_default_index(vocab['<unk>'])  
    return tokenized_dataset,vocab

def get_data(dataset, vocab, batch_size):
    data = []                                                   
    for example in dataset:
        if example['tokens']:                                      
            tokens = example['tokens'].append('<eos>')             
            tokens = [vocab[token] for token in example['tokens']] 
            data.extend(tokens)                                    
    data = torch.LongTensor(data)                                 
    num_batches = data.shape[0] // batch_size 
    data = data[:num_batches * batch_size]                       
    data = data.view(batch_size, num_batches)          
    return data
    
def get_batch(data, seq_len, num_batches, idx):
    src = data[:, idx:idx+seq_len]                   
    target = data[:, idx+1:idx+seq_len+1]             
    return src, target

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
content_path = "NLP/shakespeare.txt"
dataset,vocab = preprocess_data(content_path,content_path,content_path,tokenizer)
batch_size = 64
train_data = get_data(dataset['train'], vocab, batch_size)
val_data = get_data(dataset['val'], vocab, batch_size)
test_data = get_data(dataset['test'], vocab, batch_size)

def train(model, data, optimizer, criterion, batch_size, seq_len, clip, device,hidden_exists):
    
    epoch_loss = 0
    model.train()
    # drop all batches that are not a multiple of seq_len
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]
    if hidden_exists:
        hidden = model.init_hidden(batch_size, device)
    
    for idx in tqdm(range(0, num_batches - 1, seq_len), desc='Training: ',leave=False):  # The last batch can't be a src
        optimizer.zero_grad()
        if hidden_exists:
            hidden = model.detach_hidden(hidden)

        src, target = get_batch(data, seq_len, num_batches, idx)
        src, target = src.to(device), target.to(device)
        src.target = src.float(),target.float()
        batch_size = src.shape[0]
        if hidden_exists:
            prediction, hidden = model(src, hidden)
        else:
             prediction = model(src,target)            
        prediction = prediction.reshape(batch_size * seq_len, -1) 
        
        target = target.reshape(-1)
        if prediction.shape[1] == 1:
            prediction = torch.squeeze(prediction, 1)
        #print(f'Pred:{prediction[0]},target:{target[0]}')
        loss = criterion(prediction.float(), target.float())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() #* seq_len
        #print(epoch_loss, num_batches)
    return epoch_loss / num_batches

def evaluate(model, data, criterion, batch_size, seq_len, device,hidden_exists):

    epoch_loss = 0
    model.eval()
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]
    if hidden_exists:
        hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for idx in range(0, num_batches - 1, seq_len):
            if hidden_exists:
                hidden = model.detach_hidden(hidden)
            src, target = get_batch(data, seq_len, num_batches, idx)
            src, target = src.to(device), target.to(device)
            src.target = src.float(),target.float()
            batch_size= src.shape[0]
            if hidden_exists:
                prediction, hidden = model(src, hidden)
            else:
                prediction = model(src,target)
            prediction = prediction.reshape(batch_size * seq_len, -1)
            target = target.reshape(-1)
            if prediction.shape[1] == 1:
                prediction = torch.squeeze(prediction, 1)
            loss = criterion(prediction.float(), target.float())
            epoch_loss += loss.item() #* seq_len
    return epoch_loss / num_batches

    def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            print(src.shape)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens

    from torcheval.metrics.text import Perplexity
vocab_size = len(vocab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_dim = 1024             # 400 in the paper
hidden_dim = 1024                # 1150 in the paper
num_layers = 2                   # 3 in the paper
dropout_rate = 0.65              
tie_weights = True                  
lr = 1e-3 

n_epochs = 500
seq_len = 32
clip = 0.25
saved = False
#model = LSTM(vocab_size, embedding_dim, hidden_dim, num_layers, dropout_rate, tie_weights).to(device)
#model = Transformer(seq_len).to(device)
model = RNN(seq_len,vocab_size, 128).to(device)
hiddenexists = True #False
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss() #nn.CrossEntropyLoss()
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {num_params:,} trainable parameters')

lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)

if saved:
    model.load_state_dict(torch.load('best-val-lstm_lm.pt',  map_location=device))
    test_loss = evaluate(model, test_data, criterion, batch_size, seq_len, device)
    print(f'Test Perplexity: {math.exp(test_loss):.3f}')
else:
    best_valid_loss = torch.tensor(float('inf')).to(device)
for epoch in range(n_epochs):
    train_loss = train(model, train_data.to(device), optimizer, criterion.to(device), 
                batch_size, seq_len, clip, device,hiddenexists)
    val_loss = evaluate(model, val_data, criterion, batch_size, 
                seq_len, device,hiddenexists)
    
    lr_scheduler.step(val_loss)

    if val_loss < best_valid_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best-val-lstm_lm.pt')
    try:
        print(f'\tTrain Perplexity: {math.exp(train_loss):.3f}')
    except:
        print(f'Train Perplexity overflow, exponent: {train_loss:.3f}')
    try:
        print(f'\tVal Perplexity: {math.exp(val_loss):.3f}')
    except:
        print(f'Val Perplexity overflow, exponent: {train_loss:.3f}')

prompt = 'Think about'
max_seq_len = 50
seed = 0

temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]
for temperature in temperatures:
    generation = generate(prompt, max_seq_len, temperature, model, tokenizer, 
                          vocab, device, seed)
    print(str(temperature)+'\n'+' '.join(generation)+'\n')
