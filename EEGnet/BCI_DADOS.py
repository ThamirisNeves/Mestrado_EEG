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
from tensorflow.keras           import backend as K
from tensorflow.keras           import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils     import Sequence

import torch 

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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


class BCI():
    def __init__(self, subjects):
        """
        Initializa loading dataset
        """
        self.train_set,self.valid_set =  self.carrega_moaab(subjects)

    def carrega_moaab(self,subjects,n_jobs=7,low_cut_hz=4.0,high_cut_hz=38.0,dataset_name="BNCI2014001",offset = -0.5):
        """Load dataset"""
        dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=subjects)

        #Preprocessamento transforma os dados
        preprocessors = [
            Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Defini os tipos de sensores que serão mantidos - somente eeg
            Preprocessor(scale, factor=1e6, apply_on_array=True),         # Altera a escala dos dados
            Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),# Realiza o filtro de banda
            #Preprocessor(exponential_moving_standardize,init_block_size=init_block_size)                 # Média movel distribuida ao longo dos dados
        ]

        # Transform the data
        preprocess(concat_ds= dataset, preprocessors= preprocessors,save_dir=None,n_jobs=n_jobs)

        # Obtem a frequencia padrão dos datasets e valida se todos são iguais
        sfreq = dataset.datasets[0].raw.info['sfreq']
        assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])

        # Calcula quando começa de fato a execução do movimento.
        trial_start_offset_samples = int(offset * sfreq)

        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            preload=True,
        )


        #Divide os dados em treino e teste
        splitted = windows_dataset.split('session')
        train_set = splitted['session_T']
        valid_set = splitted['session_E']

        return (train_set,valid_set)

    def cria_datasets(self,seg=4):
        #Cria uma unica base de treino e uma unica base de teste
        base_treino = []
        for t_set in self.train_set.datasets:
            base = t_set.windows.to_data_frame()
            base['subject'] = t_set.description.subject
            base['run'] = t_set.description.run
            base_treino.append(base)
            #break
        t_df = pd.concat(base_treino).reset_index().drop('index',axis=1)

        base_valid = []
        for v_set in self.valid_set.datasets:
            base = v_set.windows.to_data_frame()
            base['subject'] = v_set.description.subject
            base['run'] = v_set.description.run
            base_valid.append(base)
            #break
        v_df = pd.concat(base_valid).reset_index().drop('index',axis=1)

        campos = [
                #'Cz','FCz','CPz'
                #'Cz','C1','C2','FCz','FC1','FC2','CPz','CP1','CP2'
                'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4','C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2','CP4', 'P1', 'Pz', 'P2', 'POz'
                ]
        base_campos = campos.copy()
        base_campos.extend(['time','condition','epoch','subject','run'])

        #utilizando eletrodos mais proximos a região central do cérebro
        df_train = t_df[base_campos].copy()
        #df_train = df_train[(df_train['condition'] == 'left_hand') | (df_train['condition'] == 'tongue')].reset_index().drop('index',axis=1)

        df_test = v_df[base_campos].copy()
        #df_test = df_test[(df_test['condition'] == 'left_hand') | (df_test['condition'] == 'tongue')].reset_index().drop('index',axis=1)

        tempo = (1125 * seg)

        #cria agregação do tempo a cada 2 segundos
        df_train['seg_agr'] = df_train['time'].apply(lambda x: (x+4)//tempo)
        #cria agregação do tempo a cada 2 segundos

        df_test['seg_agr'] = df_test['time'].apply(lambda x: (x+4)//tempo)

        return (df_train, df_test)

    def geraBase(self,df,campos,s_len,g_len):
        dit = {'left_hand':0,'tongue':1,'right_hand':2,'feet':3}
        dados = []
        y = []
        dit = {'left_hand':1,'tongue':2,'right_hand':3,'feet':4}
        for gr, df_g in tqdm.tqdm(df.groupby(by=['subject','run','epoch','condition','seg_agr'])):
            df_s = df_g.reset_index()
            max = len(df_s)
            if(max>=s_len):
                for i in range(g_len):
                    sp = df_s.sample(s_len).sort_values(by=['index'])
                    dados.append(np.array(sp[campos].to_numpy().T))
                    y.append(dit[gr[3]])
        
        y = np.array(y)
        X = np.array(dados)
        del(dados)
        
        return (X,y)

    def train_val_test(self,df_train,df_test,campos,s_len , g_len):
        X_train, Y_train  = self.geraBase(df_train,campos, s_len, g_len)
        X_train,X_validate, Y_train,Y_validate  = train_test_split(X_train,Y_train, test_size=0.33,random_state=42,stratify=Y_train)

        X_test,Y_test = self.geraBase(df_test,campos,s_len, g_len=1)

        kernels, chans, samples = 1, X_train.shape[1], X_train.shape[2]

        Y_train      = np_utils.to_categorical(Y_train-1)
        Y_validate   = np_utils.to_categorical(Y_validate-1)
        Y_test       = np_utils.to_categorical(Y_test-1)

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

    def train_NN(self,X_train,Y_train,X_validate,Y_validate,X_test,Y_test,chans, samples, kernLength = 64, F1 = 16, D = 2, F2 = 32,nb_classes = 4,batch_size = 64,epochs = 300,model_name='model.h5' ):
        
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



#bci = BCI([1,2])
#df_train, df_test = bci.cria_datasets(4)
#X,y = bci.geraBase(df_train,['Cz','FCz'],1124,1)

#288 Capturas #Sensores(Cz) = 1 e #Tamanho da Amostra


#X.shape

#X[0].shape


sujeitos = [[2]]#,[2],[3],[4],[5],[6],[7],[8],[9]]
segundos = [4]#[4,2,1,0.5]
batch_sizes = [128]#[512]#,128]
modelo   = [(64,16,2,32)]
params = {
    #0.5: [(140,1),(140,10),(140,20),(140,30),(100,10),(100,20),(100,30),(100,40),(100,60)],
    # 1: [(281,1),(281,10),(281,20),(281,30),(250,10),(250,20),(250,30),(250,40),(250,50),(250,60)],
    # 2: [(562,1),(562,10),(562,20),(562,30),(500,10),(500,20),(500,30),(500,40),(500,50),(500,60)],
    4: [(1124,1)]#,(1000,30),(1124,30)]#,(1124,30),(1000,30),(1124,40),(1000,40)]
}
path = ''#'trials\\exp3\\'
import logging
import time
import json
logging.basicConfig(filename=path+'models.log', encoding='utf-8', level=logging.DEBUG)
campos = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4']#,'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2','CP4', 'P1', 'Pz', 'P2', 'POz']



for suj in sujeitos:
    start_time_all = time.time()
    bci = BCI(suj)
    for seg in segundos:
        df_train, df_test = bci.cria_datasets(seg)
        for param in params[seg]:
            for kernLength, F1, D , F2 in modelo:
                for batch in batch_sizes:
                    X_train,Y_train,X_validate,Y_validate,X_test,Y_test,chans,samples = bci.train_val_test(df_train,df_test,campos,s_len=param[0],g_len=param[1])
                    model_name='V2model'+str(suj)+'_'+str(seg)+'_'+str(param[0])+'_'+str(param[1])+'_'+str(kernLength)+'_'+str(batch)+'_'+str(F1)+'_'+str(D)+'_'+str(F2)+'.h5' 
                    start_time = time.time()
                    name,acc,model = bci.train_NN(X_train,Y_train,X_validate,Y_validate,X_test,Y_test,chans, samples,kernLength=kernLength, F1=F1, D=D , F2=F2,batch_size = batch,epochs = 300,model_name=path+model_name )
                    end_time = time.time()
                    time_elapsed = (end_time - start_time)
                    time_elapsed_all = (end_time - start_time_all)
                    
                    #with open(path+'history\\'+model_name+".json", "w") as outfile:
                    #    json.dump(model.history, outfile)

                    logging.info(name+';'+str(acc)+';'+str(suj)+';'+str(seg)+';'+str(param[0])+';'+str(param[1])+';'+str(time_elapsed)+';'+str(time_elapsed_all)+';'+str(kernLength)+';'+str(batch)+';'+str(F1)+';'+str(D)+';'+str(F2))
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
    del(bci)

logging.shutdown()

