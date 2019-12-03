#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import argparse
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from utils_seg import getSelectedIOdata, load, createDir
from transferlearning import findnewestfile

def loadmodel(modelpath):
    model = load_model(modelpath)
    return model

class predict:

    def __init__(self, Input, save_model_dir, save_result_dir):
        self.Input = np.array(Input)
        self.save_model_dir = save_model_dir
        self.save_result_dir= save_result_dir

    def predictmodel(self):
        Predict_X = self.Input
        print('the shape of predict x is {0}'.format(Predict_X.shape))
        save_model_dir = self.save_model_dir
        save_result_dir = self.save_result_dir
        predict_res = [[] for i in range(50)]
        predict_score = [[] for i in range(50)]
        print(len(predict_res))
        model = loadmodel(save_model_dir)
        for layer in model.layers:
            layer.trainable = False    
        model.summary()
        print('========doing predicting......========')
        num=0
        for x in Predict_X:
            num += 1
            if num%1000 ==0:
                print(num)
            x = np.reshape(x, (1, x.shape[0], x.shape[1]))
            prediction = model.predict(x, verbose = 0)
            for i,d in enumerate(np.arange(0.0,1,0.02)):
                d = round(d,2)                
                if prediction[0][1] > d:
                    index = 1
                else:
                    index = 0
                predict_res[i].append(index)
                predict_score[i].append(prediction[0][1])
        print('the length of the predict result is {0}'.format(len(predict_res)))
        res_length = len(predict_res)
        return res_length,predict_score,predict_res

class test:

    def __init__(self, Input,save_model_dir):
        self.Input = np.array(Input)
        self.save_model_dir = save_model_dir

    def predicttest(self):
        Predict_X = self.Input
        print('the shape of predict x is {0}'.format(Predict_X.shape))
        save_model_dir = self.save_model_dir
        predict_res = []
        model = loadmodel(save_model_dir)
        for layer in model.layers:
            layer.trainable = False    
        model.summary()
        print('========doing predicting......========')
        for x in Predict_X:
            x = np.reshape(x, (1, x.shape[0], x.shape[1]))
            prediction = model.predict(x, verbose = 0)
            index = prediction[0][1] # return of argmax is index of the max number
            predict_res.append(index)
        print('the length of the predict result is {0}'.format(len(predict_res)))
        return predict_res
    
class performance:

    def __init__(self, window_size,predict_res_dir, label_list, IP_Chunk_Info, record_dir):
        self.predict_res_dir = predict_res_dir
        self.label_list = label_list
        self.record_dir = record_dir
        self.window_size = window_size
        self.IP_Chunk_Info = IP_Chunk_Info

    def performancedisplay(self):
        predic_list = []
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        predict_file = self.predict_res_dir
        label_list = self.label_list
        window_size = self.window_size
        record_dir = self.record_dir
        IP_Chunk_Info = self.IP_Chunk_Info
        with open(predict_file, 'r') as pf:
            while True:
                line = pf.readline()
                if line:
                    predic_list.append(int(line.strip()))
                else:
                    break
        print('=' * 20, 'RESULT', '=' * 20)
        if len(label_list) == len(predic_list):
            for i in range(len(label_list)):
                label_list[i] = int(label_list[i])
            precision, recall, f1_score, _ = np.array(list(precision_recall_fscore_support(label_list, predic_list, average = 'binary')))
            print('=' * 20, 'RESULT', '=' * 20)
            FP_list = []
            FN_list = []
            for i,(a,b) in enumerate(zip(label_list, predic_list)):
                if int(a) == 1 and int(b) == 1:
                    tp += 1
                if int(a) == 1 and int(b) == 0:
                    fn += 1
                    FN_list.append(str(i) + ' ' + str(fn) + ' ' + '(' + str(IP_Chunk_Info[i][0]) + ',' + str(IP_Chunk_Info[i][1]) + ')')
                if int(a) == 0 and int(b) == 0:
                    tn += 1
                if int(a) == 0 and int(b) == 1:
                    fp += 1
                    FP_list.append(str(i) + ' ' + str(fp) + ' ' + '(' + str(IP_Chunk_Info[i][0]) + ',' + str(IP_Chunk_Info[i][1]) + ')')
            with open(record_dir, 'w') as f:
                f.write('total fp is :' + str(fp) + '\n')
                for i in range(len(FP_list)):
                    f.write(FP_list[i] + '\n')
                f.write('total fn is :' + str(fn) + '\n')
                for j in range(len(FN_list)):
                    f.write(FN_list[j] + '\n')
            print("Precision:  %.6f, Recall: %.6f, F1_score: %.6f" % (precision, recall, f1_score))
            print('recorded! tp = {0}, fn = {1}, tn = {2}, fp = {3}'.format(tp,fn,tn,fp))
        else:
            print("the length of the file is not equal! check again")
            print(len(predic_list))
        return precision, recall, f1_score

if __name__ == '__main__':
    import tensorflow as tf
    import keras.backend.tensorflow_backend as KTF

    config = tf.ConfigProto() # configure to use 30% of GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.Session(config=config)
    KTF.set_session(session )
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_step', help='single_step.', type=int, default=1)
    parser.add_argument('--window_size', help='window_size.', type=int, default= 10)
    parser.add_argument('--rootdir', help='rootdir.', type=str, default='../Logs/')
    parser.add_argument('--label_dir', help='label_dir.', type=str, default='../labels/B6220/')
    parser.add_argument('--vector_path', help='vector_path.', type=str, default='../Logs/template_vec.dat')
    parser.add_argument('--selected_switchid_list', help='selected_switchid_list.', type=list, default=[])
    parser.add_argument('--save_model_dir', help='save_model_dir.', type=str, default='../checkpoint/transfer/')
    parser.add_argument('--save_result_dir', help='save_result_dir.', type=str, default='../result/')
    parser.add_argument('--mode', help='mode.', type=str, default='predict')
    args = parser.parse_args()
    para_predict = {
        'single_step': args.single_step,
        'window_size': args.window_size,
        'rootdir':args.rootdir,
        'label_dir': args.label_dir,
        'vector_path':args.vector_path,
        'selected_switchid_list':args.selected_switchid_list
        }
    if args.mode == 'predict':
        save_model_dir = args.save_model_dir
        save_result_dir = args.save_result_dir
        modelfile = save_model_dir+"model-transfer_LSTM128_step5_batchSize32_fixed.h5"
        save_result_path = save_result_dir + "predicRes.txt"
        predic_X, predict_Y, IP_Chunk = load(para_predict)
        res,score,predict = predict(predic_X, modelfile, save_result_path).predictmodel()
        for i,s in enumerate(score):
            with open(os.path.join(save_result_dir,'score_'+str(round(i*0.02+0.0,2))),'w') as w_score:
                for tem in s:
                    w_score.write(str(tem)+'\n')   
        for i,s in enumerate(predict):
            with open(os.path.join(save_result_dir,'predict_'+str(round(i*0.02+0.0,2))),'w') as w_score:
                for tem in s:
                    w_score.write(str(tem)+'\n')
        print('prediction down!')
    elif args.mode == 'performance':
        window_size = args.window_size
        record_dir = '../record/'
        predic_X, predict_Y, IP_Chunk = load(para_predict)
        with open(os.path.join(record_dir,'f1'),'w') as w:
            for i in range(50):
                tem = round(i*0.02+0.0,2)
                path = os.path.join(args.save_result_dir,'predict_'+str(round(i*0.02+0.0,2)))
                recordpath = record_dir + 'record_' + str(round(i*0.02+0.0,2))
                precision, recall, f1_score = performance(window_size,path,predict_Y,IP_Chunk, recordpath).performancedisplay()
                f1 = str(tem) + ' ' + str(precision)+ ' ' + str(recall)+ ' ' + str(f1_score)
                w.write(f1+'\n')
        i = []
        p = []
        r = []
        f = []
        with open(os.path.join(record_dir,'f1'),'r') as r_f1:
            for l in r_f1:
                l=l.split()
                i.append(float(l[0]))
                p.append(float(l[1]))
                r.append(float(l[2]))
                f.append(float(l[3]))
                with open('p_r_record.txt','a') as f_pr:
                    f_pr.write(str(float(l[2])) + ' ' + str(float(l[1])) + '\n')
        i = np.array(i)
        p = np.array(p)
        r = np.array(r)
        f = np.array(f)
        best = np.argmax(f)
        plt.figure(1)
        prf = plt.subplot(2,1,1)
        plt.sca(prf)
        plt.plot(i,p,color='red')       
        plt.plot(i,r,color='blue')  
        plt.plot(i,f,color='green') 
        plt.scatter(round(best*0.02+0.0,2),f[best])
        plt.xlabel('threshold')
        plt.ylabel('score')
        pr = plt.subplot(2,1,2)
        plt.sca(pr)
        plt.plot(p,r,color='red')       
        plt.xlabel('p')
        plt.ylabel('r')
        print(f[best])
        plt.show()
        print('performancedisplay down!')
    elif args.mode == 'test':
        predic_X, predict_Y, IP_Chunk = load(para_predict)
        int_pre_Y  = list(map(int,predict_Y))
        index = []
        for i,ele in enumerate(int_pre_Y):
            if ele ==1:
                index.append(i)
        with open('../testresult2.txt','r') as r:
            score = r.readlines()[0][1:-1]
        int_score  = list(map(float,score.split(',')))
        plot_wrong = []
        plot_right = []
        for i,ele in enumerate(int_score):
            if i in index:
                plot_wrong.append(ele)
            else:
                plot_right.append(ele)
        length = len(int_score)
        x = [ i for i in range(length)]
        index_right = list(set(x)-set(index))
        plt.scatter(index,plot_wrong,c='green')
        plt.scatter(index_right,plot_right,c='red',alpha='0.1') 
        plt.plot(x,[0.5]*length)
        plt.show()
