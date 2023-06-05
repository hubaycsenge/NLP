import gc
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm

def train(model,optimizer,criterion,dataloader,epoch,device,hidden_exists = False,n_hiddens = 2,hidden = None,verbose = False):
    epoch_loss = 0
    idx = 0
    
    if hidden_exists:
        if hidden is None:
            h = torch.rand(1, 1,300).to(device)
            hidden = [h]* n_hiddens
            if n_hiddens == 1:
                hidden = hidden[0]
        
        #hidden = model.init_hidden(1, device)
    for prompt,target in tqdm(dataloader):
        #if hidden_exists:
            #hidden = model.detach_hidden(hidden)
        prompt, target = prompt.to(device), target.to(device)
        if hidden_exists:
            prediction, hidden = model(prompt, hidden)
            if n_hiddens>1:
                hidden = [hidden[i].detach()for i in range(n_hiddens)]
            else:
                hidden = hidden.detach()
        else:
            prediction = model(prompt,target)
        try:
            loss = criterion(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            #nn.utils.clip_grad_value_(model.parameters(), 0.25)
            optimizer.step()
            lossit = loss.item()
        except:
            def closure():
                optimizer.zero_grad()
                loss = criterion(prediction, target)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
        
        if idx%1000==0:
            if verbose:
                print(lossit)
            '''print('TARGET')
            print(target[0,0,:25])
            print('HIDDEN')
            print('shape:',hidden.shape,prediction.shape)
            print(hidden[0,0,:25])
            print('PREDICTION')
            print(prediction[0,0,:25])'''
        epoch_loss += lossit
        
        idx += 1   
            
        
    print(f'Epoch #{epoch + 1} train loss: {epoch_loss/len(dataloader)}')
    return hidden,(epoch_loss/len(dataloader))

def evaluate(model,criterion,dataloader,epoch,device,hidden_exists = False,n_hiddens = 2,hidden = None):
    if hidden_exists:
        if hidden is None:
            h = torch.randn(1, 1,300).to(device)
            hidden = [h]* n_hiddens
            if n_hiddens == 1:
                hidden = hidden[0]
    epoch_loss = 0
    for prompt,target in tqdm(dataloader):
        
        prompt, target = prompt.to(device), target.to(device)
        if hidden_exists:
            prediction, hidden = model(prompt, hidden)
            if n_hiddens>1:
                hidden = [hidden[i].detach()for i in range(n_hiddens)]
            else:
                hidden = hidden.detach()
        else:
            prediction = model(prompt,target)  
        loss = criterion(prediction, target,torch.ones(1))    
        epoch_loss += loss.item()
    print(f'Epoch #{epoch + 1} val loss: {epoch_loss/(len(dataloader))}')
    return(epoch_loss/len(dataloader))

def evaluate_with_translate(model,w2v_model,criterion,dataloader,epoch,device,hidden_exists = False,n_hiddens = 2,hidden = None,verbose = False):
    if hidden_exists:
        if hidden is None:
            h = torch.randn(1, 1,300).to(device)
            hidden = [h]* n_hiddens
            if n_hiddens == 1:
                hidden = hidden[0]
    epoch_loss = 0
    idx = 0
    for prompt,target in tqdm(dataloader):
        
        prompt, target = prompt.to(device), target.to(device)
        if hidden_exists:
            prediction, hidden = model(prompt, hidden)
            if n_hiddens>1:
                hidden = [hidden[i].detach()for i in range(n_hiddens)]
            else:
                hidden = hidden.detach()
        else:
            prediction = model(prompt,target)  
        pred_squeezed = torch.squeeze(prediction,0)
        target_squeezed = torch.squeeze(target,0)
        info = (torch.ones(prediction.shape[1])*-1).to(device)

        loss = criterion(pred_squeezed,target_squeezed,info)
        epoch_loss += loss.item()
        if idx %1000 == 0:
            if verbose:
                pred_words = []
                target_words = []
                for i in range(prediction.shape[1]):
                    prediction_word = w2v_model.similar_by_vector(np.array(prediction[0,i,:].cpu().detach()),topn=1)[0][0]
                    target_word = w2v_model.similar_by_vector(np.array(target[0,i,:].cpu().detach()),topn=1)[0][0]
                    pred_words.append(prediction_word)
                    target_words.append(target_word)
                pred_sentence = ' '.join(pred_words)
                target_sentence = ' '.join(target_words)
                print(f'##########PREDICTED##########: {pred_sentence}, \n ##########TARGET##########: {target_sentence}')
    print(f'Epoch #{epoch + 1} val loss: {epoch_loss/(len(dataloader))}')
    return(epoch_loss/len(dataloader))

def train_with_translate(model,w2v_model,optimizer,criterion,dataloader,epoch,device,hidden_exists = False,n_hiddens = 2,hidden = None,verbose = False):
    epoch_loss = 0
    idx = 0
    loss_dict = {}
    if hidden_exists:
        if hidden is None:
            h = torch.rand(1, 1,300).to(device)
            hidden = [h]* n_hiddens
            if n_hiddens == 1:
                hidden = hidden[0]
        
        #hidden = model.init_hidden(1, device)
    
    for prompt,target in tqdm(dataloader):
        #if hidden_exists:
            #hidden = model.detach_hidden(hidden)
        prompt, target = prompt.to(device), target.to(device)
        if hidden_exists:
            prediction, hidden = model(prompt, hidden)
            if n_hiddens>1:
                hidden = [hidden[i].detach()for i in range(n_hiddens)]
            else:
                hidden = hidden.detach()
        else:
            prediction = model(prompt,target)
        
        pred_squeezed = torch.squeeze(prediction,0)
        target_squeezed = torch.squeeze(target,0)
        info = (torch.ones(prediction.shape[1])*-1).to(device)

        loss = criterion(pred_squeezed,target_squeezed,info)
        lossit = loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step() 
        
        if idx %100 == 0:
            loss_dict[idx] = float(lossit)
        if idx%1000==0:
            with  open(f"{epoch}.json", "w") as out_file:
                json.dump(loss_dict,out_file)
            if verbose:
                print(lossit)
                pred_words = []
                target_words = []
                for i in range(prediction.shape[1]):
                    prediction_word = w2v_model.similar_by_vector(np.array(prediction[0,i,:].cpu().detach()),topn=1)[0][0]
                    target_word = w2v_model.similar_by_vector(np.array(target[0,i,:].cpu().detach()),topn=1)[0][0]
                    pred_words.append(prediction_word)
                    target_words.append(target_word)
                pred_sentence = ' '.join(pred_words)
                target_sentence = ' '.join(target_words)
                print(f' \n  ##########PREDICTED##########:  \n {pred_sentence}, \n ##########TARGET##########:  \n {target_sentence}')

        epoch_loss += lossit
        
        idx += 1   
            
        
    print(f'Epoch #{epoch + 1} train loss: {epoch_loss/len(dataloader)}')
    return hidden,(epoch_loss/len(dataloader))



def prediction_step(NLP_model,prompt,hidden = None,n_hiddens = 2,target = None):
    if hidden is not None:
        prediction, hidden = NLP_model(prompt, hidden)
        if n_hiddens>1:
            hidden = [hidden[i].detach()for i in range(n_hiddens)]
        else:
            hidden = hidden.detach()
        return prediction, hidden
    else:
        prediction = NLP_model(prompt,target)
        return prediction

def prompt_to_vector(prompt,w2v_model):
    sentence_vec = []
    for word in prompt.split(' '):
        if '@' not in word and '=' not in word:
            try:
                sentence_vec.append(w2v_model[word])
            except:
                pass
                #print(f'{word} not found')
    #sentence_vec = [torch.from_numpy(item).float() for item in sentence_vec]
    if len(sentence_vec) > 0:
        sentence_vec = np.array(sentence_vec)
    else:
        raise Exception('Invalid prompt encountered')
    return torch.as_tensor(sentence_vec)

def generate_target(prompt):
    target = torch.zeros(prompt.shape)
    target[:prompt.shape[0]-1, :] = prompt[1:,:]
    return target

def update_prompt(prompt,output):
    #print(prompt.shape)
    new_prompt = torch.zeros(prompt.shape)
    #print(new_prompt[:prompt.shape[0]-1, :].shape,prompt[1:,:])
    new_prompt[:prompt.shape[0]-1, :] = prompt[1:,:]
    new_prompt[-1,:] = output
    return new_prompt
            
    

def generate(NLP_model,w2v_model,prompt,n_words,device,hidden_exists = False,n_hiddens = 2,hidden = None):
    NLP_model.eval()
    NLP_model = NLP_model.to(device)
    promptvec = prompt_to_vector(prompt,w2v_model).to(device)
    words_generated = []
    if not(hidden_exists):
        targetvec = generate_target(promptvec).to(device)
    for idx in range(n_words):
        if hidden_exists:
            prediction, hidden = prediction_step(NLP_model,promptvec.unsqueeze(0),hidden = hidden,n_hiddens = n_hiddens)
        else:
            prediction = prediction_step(NLP_model,promptvec.unsqueeze(0),target = targetvec.unsqueeze(0))
            targetvec = generate_target(promptvec)
        output = prediction[0,-1,:]
        word =  w2v_model.similar_by_vector(np.array(output.cpu().detach()),topn=1)[0][0]
        words_generated.append(word)
        promptvec = update_prompt(promptvec,output)
        
    print(f'For input {prompt}')
    print(f'The output is: {" ".join(words_generated)}')
    return words_generated
            
        
        
        
        

