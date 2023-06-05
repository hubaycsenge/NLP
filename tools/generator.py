import torch
import numpy as np
from torch.utils.data import Dataset

class NLPDataset(Dataset):
    def __init__(self, source, transform=None, target_transform=None,output_seq_len = 512,add_padding = False):
        self.source = source
        self.stopword = stopword = torch.load('stopword.pt')
        self.transform = transform
        self.target_transform = target_transform
        self.data = torch.load(source)
        self.max_sentence_len = max([sentence.shape[0] for sentence in self.data])
        self.output_seq_len = output_seq_len
        self.padding = add_padding


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = self.data[idx]
        target = self.generate_target(prompt)
        if self.padding:
            prompt,target,mask = self.pad(prompt,target,output_seq_len = self.output_seq_len)
            return prompt,target,mask
        else:
            return prompt,target

    def generate_target(self,prompt):
        target = torch.zeros(prompt.shape)
        for i in range(prompt.shape[0]):
            try:
                if i < prompt.shape[0] - 1:
                    target[i,:] = prompt[i+1,:]
                else:          
                    target[i,:] = self.stopword
            except:
                pass

        return target

    def pad(self,prompt,target,output_seq_len = 512):
        P = torch.zeros((512,300))
        T = torch.zeros((512,300))
        M = torch.zeros((512,300))
        P[:prompt.shape[0],:300, ] = torch.transpose(prompt, 0, 1)
        T[:300, :target.shape[0]] = torch.transpose(target, 0, 1)
        M[:300, :prompt.shape[0]] = torch.ones(torch.transpose(prompt, 0, 1).shape)
        return P,T,M

    def update_seq_len(self,new_seq_len):
        self.output_seq_len = new_seq_len