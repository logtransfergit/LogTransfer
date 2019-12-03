#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import keras
import argparse
import time
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Embedding, LSTM, Dropout, Bidirectional, Input, Masking, TimeDistributed, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from utils_seg import getSelectedIOdata, load

class nn:
    	
	def __init__(self, Input, Output, model_dir):
		self.Input = np.array(Input)
		self.Output = np.array(Output)
		self.model_dir = model_dir

	def trainingModel(self):
		t1 =time.time()
		batchSize = 256
		LSTMhiddenDims = 128
		DensehiddenDims = 150
		outputDims = 2
		VectorSize = 150
		WindowSize = 20
		train_X = self.Input
		train_Y = np_utils.to_categorical(self.Output, num_classes= outputDims)
		model_dir = self.model_dir
		print('the shape of train x is {0}'.format(train_X.shape))
		print('the shape of train y is {0}'.format(train_Y.shape))
		print('the model_dir is {0}'.format(model_dir))		 
		template_input = Input(shape= (WindowSize, VectorSize), dtype = 'float32', name = 'template_input')
		lstm1 = LSTM(LSTMhiddenDims, return_sequences = True)(template_input)
		lstm1_d = Dropout(0.5)(lstm1)
		lstm2 = LSTM(LSTMhiddenDims)(lstm1_d)
		lstm2_d = Dropout(0.5)(lstm2)
		dense1 = Dense(DensehiddenDims, activation = 'relu', name = 'dense1')(lstm2_d)
		output = Dense(outputDims, activation = 'softmax')(dense1)
		model = Model(inputs = [template_input], outputs = output)
		model.summary()
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		filepath = model_dir + 'log_weights-rm-{epoch:02d}-{loss:.4f}-bigger.hdf5'
		checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [checkpoint]
		model.fit(train_X, train_Y, epochs=25, batch_size=batchSize, verbose = 0, callbacks=callbacks_list)
		t2 = time.time()
		print('training time:',(t2-t1)/60,'mins')

if __name__ == '__main__':
	import tensorflow as tf
	import keras.backend.tensorflow_backend as KTF
	from keras import backend as K

	config = tf.ConfigProto() # configure to use 30% of GPU
	config.gpu_options.per_process_gpu_memory_fraction = 0.3
	session = tf.Session(config=config)
	KTF.set_session(session )
	parser = argparse.ArgumentParser()
	parser.add_argument('--single_step', help='single_step.', type=int, default=1)
	parser.add_argument('--window_size', help='window_size.', type=int, default= 10)
	parser.add_argument('--rootdir', help='rootdir.', type=str, default='../Logs')
	parser.add_argument('--label_dir', help='label_dir.', type=str, default='../labels/')
	parser.add_argument('--vector_path', help='vector_path.', type=str, default='../Logs/template_vec.dat')
	parser.add_argument('--selected_switchid_list', help='selected_switchid_list.', type=list, default=[])
	args = parser.parse_args()
	para_source = {
		'single_step' : args.single_step,
	    'window_size': args.window_size,
	    'rootdir': args.rootdir,
	    'label_dir': args.label_dir,
	    'vector_path': args.vector_path,
	    'selected_switchid_list':args.selected_switchid_list
	} 
	save_model_dir = '../checkpoint/model/'
	TrainX, TrainY, IP_Chunk = load(para_source)
	TrainLstm  = nn(TrainX, TrainY, save_model_dir).trainingModel()
	K.clear_session()
	print('template training finish')
