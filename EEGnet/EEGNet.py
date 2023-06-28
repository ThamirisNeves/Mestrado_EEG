#!/usr/bin/env python
# coding: utf-8

# ## Importanto dataset do moabb usando braindecode
# 
# 
# 
# 

# In[ ]:


get_ipython().system('pip3 install torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html')
get_ipython().system('pip install moabb braindecode')


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


from braindecode import datasets
from braindecode.datasets import MOABBDataset

subject_id = [3]
dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=subject_id)


# In[ ]:


print(dataset.description)


# ## Pré-processamento

# In[ ]:


from numpy import multiply
from braindecode.preprocessing import (
    exponential_moving_standardize, preprocess, Preprocessor, scale)

low_cut_hz = 4.  # low cut frequency for filtering
high_cut_hz = 38.  # high cut frequency for filtering
# Parameters for exponential moving standardization
factor_new = 1e-3
init_block_size = 1000

preprocessors = [
    Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Mantendo apenas os sinais de EEG
    Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convertendo V para uV
    Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    #Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
    #             factor_new=factor_new, init_block_size=init_block_size)
]

# Transform the data
preprocess(dataset, preprocessors)


# ## Criando janela de eventos

# In[ ]:


#import numpy as np
from braindecode.preprocessing import \
    create_windows_from_events, create_fixed_length_windows

trial_start_offset_seconds = -0.5
# Extract sampling frequency, check that they are same in all datasets
sfreq = dataset.datasets[0].raw.info['sfreq']
assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
# Calculate the trial start offset in samples.
trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

windows_dataset = create_windows_from_events(
    dataset,
    trial_start_offset_samples=trial_start_offset_samples,
    trial_stop_offset_samples=0,
    preload=True,
)


# In[ ]:


windows_dataset.datasets[0].windows.to_data_frame()


# ## Configurando GPU

# In[ ]:


get_ipython().system('nvidia-smi')


# In[ ]:


import torch
torch.cuda.is_available()
# Output would be True if Pytorch is using GPU otherwise it would be False.


# In[ ]:


import tensorflow as tf
tf.test.gpu_device_name()
# Standard output is '/device:GPU:0'


# ## EEGNet
# 

# """ Written by, Sriram Ravindran, sriram@ucsd.edu
# 
# Original paper - https://arxiv.org/abs/1611.08024
# 
# Please reach out to me if you spot an error. """

# In[ ]:


sensores = ['Fz','FC3', 'FCz', 'FC2', 'FC4', 'C3', 'C1', 'Cz',
       'C2','C4', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2','POz']


# In[ ]:


import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim


# In[ ]:


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        self.T = 120
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints. 
        self.fc1 = nn.Linear(4*2*7, 1)
        

    def forward(self, x):
        # Layer 1
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        x = x.permute(0, 3, 1, 2)
        
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        x = self.pooling3(x)
        
        # FC Layer
        x = x.reshape(-1, 4*2*7)
        x = F.sigmoid(self.fc1(x))
        return x


net = EEGNet().cuda(0)
print(net.forward(Variable(torch.Tensor(np.random.rand(1, 1, 120, 64)).cuda(0))))
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters())


# In[ ]:


def evaluate(model, X, Y, params = ["acc"]):
    results = []
    batch_size = 100
    
    predicted = []
    
    for i in range(len(X)/batch_size):
        s = i*batch_size
        e = i*batch_size+batch_size
        
        inputs = Variable(torch.from_numpy(X[s:e]).cuda(0))
        pred = model(inputs)
        
        predicted.append(pred.data.cpu().numpy())
        
        
    inputs = Variable(torch.from_numpy(X).cuda(0))
    predicted = model(inputs)
    
    predicted = predicted.data.cpu().numpy()
    
    for param in params:
        if param == 'acc':
            results.append(accuracy_score(Y, np.round(predicted)))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2*precision*recall/ (precision+recall))
    return results


# In[ ]:


X_train = treino[sensores].to_numpy()
y_train = treino['condition'].to_numpy()

X_val = teste[sensores].to_numpy()
y_val = teste['condition'].to_numpy()

X_test = teste[sensores].to_numpy()
y_test = teste['condition'].to_numpy


# In[ ]:


len(X_train)


# In[ ]:


y_train


# In[ ]:


X_train


# In[ ]:


batch_size = 32

for epoch in range(10):  # loop over the dataset multiple times
    print("\nEpoch ", epoch)
    
    running_loss = 0.0
    for i in range(len(X_train)//batch_size-1):
        s = i*batch_size
        e = i*batch_size+batch_size
        
        inputs = torch.from_numpy(X_train[s:e])
        labels = torch.FloatTensor(np.array([y_train[s:e]]).T*1.0)
        
        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda(0)), Variable(labels.cuda(0))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        
        optimizer.step()
        
        running_loss += loss.data[0]
    
    # Validation accuracy
    params = ["acc", "auc", "fmeasure"]
    print(params)
    print("Training Loss ", running_loss)
    print("Train - ", evaluate(net, X_train, y_train, params))
    print("Validation - ", evaluate(net, X_val, y_val, params))
    print("Test - ", evaluate(net, X_test, y_test, params))


# In[ ]:





# #Explicadores de modelo caixa preta
# 

# ##PDP

# In[ ]:





# ##ALE

# In[ ]:





# ##Protótipos

# In[ ]:





# ##ICE
# 

# In[ ]:





# ##Counterfactual
# 

# In[ ]:





# ##SHAP ou IG

# In[ ]:




