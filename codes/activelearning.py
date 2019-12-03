#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import keras
import argparse
import time
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout,Bidirectional, Input, Masking, TimeDistributed, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from utils import getSelectedIOdata, load

def loadmodel(modelpath):
    model = load_model(modelpath)
    return model

class nn:

    def __init__(self, Input, Output, model_dir):
        self.Input = np.array(Input)
        self.Output = np.array(Output)
        self.model_dir = model_dir

    def trainingModel(self):
        t1 =time.time()
        batchSize = 64
        LSTMhiddenDims = 128
        DensehiddenDims = 50
        outputDims = 2
        VectorSize = 100
        WindowSize = 10
        train_X = self.Input
        train_Y = np_utils.to_categorical(self.Output, num_classes= outputDims)
        model_dir = self.model_dir
        print('the shape of train x is {0}'.format(train_X.shape))
        print('the shape of train y is {0}'.format(train_Y.shape))
        print('the model_dir is {0}'.format(model_dir))     
        template_input = Input(shape= (WindowSize, VectorSize), dtype = 'float32', name = 'template_input')
        lstm1 = LSTM(LSTMhiddenDims, return_sequences = True)(template_input)
        lstm1_d = Dropout(0.2)(lstm1)
        lstm2 = LSTM(LSTMhiddenDims)(lstm1_d)
        lstm2_d = Dropout(0.2)(lstm2)
        dense1 = Dense(DensehiddenDims, activation = 'relu', name = 'dense1')(lstm2_d)
        output = Dense(outputDims, activation = 'softmax')(dense1)
        model = Model(inputs = [template_input], outputs = output)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        filepath = model_dir + 'test'
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]
        model.fit(train_X, train_Y, epochs=20, batch_size=128, callbacks=callbacks_list)
        t2 = time.time()
        print('training time:',(t2-t1)/60,'mins')

class predict:

    def __init__(self, Input, IP_Chunk, save_model_dir,mode,lc_size):
        self.Input = np.array(Input)
        self.save_model_dir = save_model_dir
        self.mode = mode
        self.lc_size = lc_size
        self.IP_Chunk = IP_Chunk

    def predictmodel(self):
        IP_Chunk = self.IP_Chunk
        Predict_X = self.Input
        print('the shape of predict x is {0}'.format(Predict_X.shape))
        save_model_dir = self.save_model_dir
        predict_res = []
        not_sure_label_x = []
        not_sure_index = []
        model = loadmodel(save_model_dir)
        for layer in model.layers:
            layer.trainable = False    
        model.summary()
        print('========doing predicting......========')
        if self.mode == 'lc':
            prediction_res = []
            max_prediction = []
            max_index = []
            for i,x in enumerate(Predict_X):
                x = np.reshape(x, (1, x.shape[0], x.shape[1]))        
                prediction = model.predict(x, verbose = 0)
                max_prediction.append(np.max(prediction))
                max_index.append(i)
                predict_res.append(prediction)
            min_pro = dict(zip(max_prediction,max_index))
            if self.lc_size > len(min_pro):
                return [],[]
            sort_min_pro=sorted(min_pro)[:self.lc_size]
            sort_min_pro_index = [min_pro[j] for j in sort_min_pro]
            for p,m in enumerate(sort_min_pro_index):
                not_sure_label_x.append(Predict_X[m])
                not_sure_index.append([m,IP_Chunk[m],sort_min_pro[p],predict_res[m]])
        elif self.mode == 'bt':
            for i,x in enumerate(Predict_X):
                x = np.reshape(x, (1, x.shape[0], x.shape[1]))        
                prediction = model.predict(x, verbose = 0)
                prediction_bt = abs(prediction[0]-prediction[1])
                all_prediction.append(prediction_bt)
                all_index.append(i)
            all_pro = dict(zip(all_prediction,all_index))
            if self.bt_size > len(all_pro):
                return [],[]
            sort_min_pro=sorted(all_pro)[:self.bt_size]
            sort_min_pro_index = [min_pro[j] for j in sort_min_pro]
            for p,m in enumerate(sort_min_pro_index):
                not_sure_label_x.append(Predict_X[m])
                not_sure_index.append([m,IP_Chunk[m],sort_min_pro[p]])
        else:
            max_prediction = []
            max_index = []
            for i,x in enumerate(Predict_X):
                x = np.reshape(x, (1, x.shape[0], x.shape[1]))        
                prediction = model.predict(x, verbose = 0)
                prediction_bt = abs(prediction[0]-prediction[1])
                all_prediction.append(prediction_bt)
                all_index.append(i)
                max_prediction.append(np.max(prediction))
                max_index.append(i)
            max_pro = dict(zip(max_prediction,max_index))
            if self.lc_size > len(max_pro):
                return [],[]
            sort_max_pro=sorted(max_pro)[:self.lc_size]
            sort_max_pro_index = [max_pro[j] for j in sort_max_pro]
            for p,m in enumerate(sort_max_pro_index):
                not_sure_label_x.append(Predict_X[m])
                not_sure_index.append([m,IP_Chunk[m],sort_max_pro[p]])
            all_pro = dict(zip(all_prediction,all_index))
            if self.bt_size > len(all_pro):
                return [],[]
            sort_min_pro=sorted(all_pro)[:self.bt_size]
            sort_min_pro_index = [min_pro[k] for k in sort_min_pro]
            for q,n in enumerate(sort_min_pro_index):
                not_sure_label_x.append(Predict_X[n])
                not_sure_index.append([n,IP_Chunk[n],sort_min_pro[q]])
        return not_sure_label_x ,not_sure_index
    
if __name__ == '__main__':
    from keras import backend as K

    parser = argparse.ArgumentParser()
    parser.add_argument('--single_step', help='single_step.', type=int, default=1)
    parser.add_argument('--window_size', help='window_size.', type=int, default= 10)
    parser.add_argument('--rootdir', help='rootdir.', type=str, default='../Logs')
    parser.add_argument('--label_dir', help='label_dir.', type=str, default='../labels/D2020/')
    parser.add_argument('--vector_path', help='vector_path.', type=str, default='../Logs/template_vec.dat')
    parser.add_argument('--selected_switchid_list', help='selected_switchid_list.', type=list, default=[])
    parser.add_argument('--mode', help='mode.', type=str, default='lc') # lc bt both
    parser.add_argument('--lc_size', help='lc_size.', type=int, default=1000)
    parser.add_argument('--bt_size', help='bt_size.', type=int, default=1000)
    parser.add_argument('--save_result_dir', help='save_result_dir.', type=str, default='../result/activelearning1.txt')
    args = parser.parse_args()
    para_source = {
        'single_step' : args.single_step,
        'window_size': args.window_size,
        'rootdir': args.rootdir,
        'label_dir': args.label_dir,
        'vector_path': args.vector_path,
        'switch_start': args.switch_start,
        'switch_end': args.switch_end
    } 
    save_model_dir = '../checkpoint/model/'
    TrainX_origin, TrainY_origin, IP_Chunk = load(para_source)
    train_id = sorted(np.random.choice(len(TrainY_origin),int(0.3*len(TrainY_origin)),replace=False))
    all_id = set(range(len(TrainY_origin)))
    al_id = list(all_id-set(train_id))
    TrainX = [TrainX_origin[i] for i in train_id]
    TrainY = [TrainY_origin[i] for i in train_id]
    alX = [TrainX_origin[i] for i in al_id]
    TrainLstm  = nn(TrainX, TrainY, save_model_dir).trainingModel()
    modelfile = save_model_dir+"test"
    mode = args.mode
    lc_size = args.lc_size
    bt_size = args.bt_size
    res, index = predict(alX, IP_Chunk, modelfile, mode, lc_size).predictmodel()
    with open(args.save_result_dir,'w') as w:
        for i in range(len(res)):
            w.write(str(index[i]) + '\n')
    K.clear_session()
    print('template training finish')
