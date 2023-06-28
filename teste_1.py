#!/usr/bin/env python
# coding: utf-8

import mne
import numpy as np
import os
import pandas as pd
import sys
import tqdm

#Caminho é o local onde se encontra o arquivo EEGModels.py

caminho = '/home/tneves/Documents/Projeto_mestrado/Projeto_mestrado/Codigos/EEGnet'

sys.path.append(os.path.abspath(caminho))
 
from braindecode.preprocessing  import create_windows_from_events
from braindecode.preprocessing  import exponential_moving_standardize
from braindecode.datasets       import MOABBDataset
from braindecode.preprocessing  import preprocess
from braindecode.preprocessing  import Preprocessor
from braindecode.preprocessing  import scale
from EEGModels                  import EEGNet
from matplotlib                 import pyplot as plt
from mne                        import io
from mne.datasets               import sample
from pyriemann.estimation       import XdawnCovariances
from pyriemann.tangentspace     import TangentSpace
from pyriemann.utils.viz        import plot_confusion_matrix
from sklearn.linear_model       import LogisticRegression
from sklearn.model_selection    import train_test_split
from sklearn.pipeline           import make_pipeline
from sklearn.preprocessing      import normalize
import tensorflow as tf
from tensorflow.keras           import backend as K
from tensorflow.keras           import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils     import Sequence


#gpus = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(gpus[0], True)

#pycol - parquet
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch 

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = 'cuda' if cuda else 'cpu'
if cuda:
    torch.backends.cudnn.benchmark = False


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
    
    
def train_val_test(train_set,valid_set, ytrain, yvalid):
    X_train = train_set
    Y_train = ytrain
    X_test = valid_set
    Y_test = yvalid
    #X_train, Y_train  = self.carrega_moaab(1)
    X_train,X_validate, Y_train,Y_validate  = train_test_split(X_train,Y_train, test_size=0.33,random_state=42,stratify=Y_train)

    #X_test,Y_test = self.geraBase(df_test,campos,s_len, g_len=1)

    kernels, chans, samples = 1, X_train.shape[1], X_train.shape[2]

    Y_train      = np_utils.to_categorical(Y_train)
    Y_validate   = np_utils.to_categorical(Y_validate)
    Y_test       = np_utils.to_categorical(Y_test
                                           )

    X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
    X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
    X_test       = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], kernels)
    
    print(X_train.shape)
    print(X_validate.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_validate.shape)
    print(Y_test.shape)

    return (X_train,Y_train,X_validate,Y_validate,X_test,Y_test,chans, samples)

def train_NN(X_train,Y_train,X_validate,Y_validate,X_test,Y_test,chans, samples, kernLength = 64, F1 = 16, D = 2, F2 = 32,nb_classes = 4,batch_size = 64,epochs = 300,model_name='model.h5' ):
        
    train_gen = DataGenerator(X_train, Y_train, batch_size)
    test_gen = DataGenerator(X_test, Y_test, batch_size)
    val_gen = DataGenerator(X_validate, Y_validate, batch_size)
    
    model = EEGNet(nb_classes = nb_classes, Chans = chans, Samples = samples, 
                dropoutRate = 0.5, kernLength = kernLength, F1 = F1, D = D, F2 = F2, 
                dropoutType = 'Dropout')

    checkpointer = ModelCheckpoint(filepath=model_name, verbose=1,
                                save_best_only=True)

    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                metrics = ['accuracy'])

    m_fit = model.fit(train_gen,
                    batch_size = batch_size,
                    epochs = epochs, 
                    validation_data=val_gen,
                    callbacks=[checkpointer],
                    )

    # load optimal weights
    model.load_weights(model_name)

    probs       = model.predict(test_gen)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == Y_test.argmax(axis=-1))
    return (model_name,acc,m_fit)


suj = str(XX)
path = '/home/tneves/Documents/Projeto_mestrado/Projeto_mestrado/Codigos/AMUSE_classe_sujeito_'+suj+'/'

"""Load dataset"""
train_set = np.load(path+'/suj_'+suj+'_train.npy')
valid_set = np.load(path+'/suj_'+suj+'_valid.npy')
Ytrain = np.load(path+'/suj_'+suj+'_y_train.npy')
Yvalid = np.load(path+'/suj_'+suj+'_y_valid.npy')

print(train_set.shape)
print(Yvalid.shape)


import logging
import time
import json
logging.basicConfig(filename=path+'models.log', encoding='utf-8', level=logging.DEBUG)


segundos = [4]#[4,2,1,0.5]
batch_sizes = [128]#[512]#,128]
modelo   = [(64,16,2,32)]

start_time_all = time.time()
df_train, df_test, ytrain, yvalid = train_set, valid_set, Ytrain, Yvalid
for kernLength, F1, D , F2 in modelo:
    for batch in batch_sizes:
        X_train,Y_train,X_validate,Y_validate,X_test,Y_test,chans,samples = train_val_test(df_train,df_test,ytrain, yvalid)
        model_name='V2model'+str(suj)+'_'+str(kernLength)+'_'+str(batch)+'_'+str(F1)+'_'+str(D)+'_'+str(F2)+'.h5' 
        start_time = time.time()
        name,acc,model = train_NN(X_train,Y_train,X_validate,Y_validate,X_test,Y_test,chans, samples,kernLength=kernLength, F1=F1, D=D , F2=F2,batch_size = batch,epochs = 300,model_name=path+model_name)
        end_time = time.time()
        time_elapsed = (end_time - start_time)
        time_elapsed_all = (end_time - start_time_all)
        with open(path+'suj_' + str(suj) + '_history.txt','w') as f:
            f.write(str(model.history))
        #with open(path+'history\\'+model_name+".json", "w") as outfile:
        #    json.dump(model.history, outfile)
        logging.info(name+';'+str(acc)+';'+str(suj)+';'+str(time_elapsed)+';'+str(time_elapsed_all)+';'+str(kernLength)+';'+str(batch)+';'+str(F1)+';'+str(D)+';'+str(F2))
        del(X_train)
        del(Y_train)
        del(X_validate)
        del(Y_validate)
        del(X_test)
        del(Y_test)
        del(chans)
        del(samples)
del(df_train)
del(df_test)
del(ytrain)
del(yvalid)
logging.shutdown()

