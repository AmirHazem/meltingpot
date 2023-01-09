# !/usr/bin/python
# -*- coding: utf-8 -*-

# Author: Amir HAZEM
# DATE: 01/12/2022
# Comment: This code takes as input a sequence of integers and return their product
# Using FNN, Simple RNN, LSTM or BiLSTM

from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense, Bidirectional, SimpleRNN, Flatten
import numpy as np
import sys
import random
import argparse
#-----------------------------------------
# Functions
#-----------------------------------------

# Generates training data
#	train_size: corresponds to the number of training sequences of length seq_length
#	seq_length: sequence size
#	max_int: integer values between 0 and max_int
#	For instance:
#		train_size = 20
#		seq_length = 2 
#		corresponds to 20 training sequences of length 2
# 		this code can be extended to multiply more than two numbers if the sequence is longer

def gen_train_data(train_size, seq_length, max_int):

	x = list()
	y = list()
	
	for i in range(0,train_size):
		X1 = [(random.randint(0, max_int)) for j in range(0, seq_length)]
		x += X1
		y.append(np.prod(X1))
	
	X = np.array(x).reshape(train_size, seq_length, 1).astype(np.float64)
	Y = np.array(y).reshape(train_size, 1, 1).astype(np.float64)
	

	return(X, Y)


# Train FNN
def train_fnn(X, Y, hidden_size, seq_length, numb_features, epochs, batch_size):
	model = Sequential()
	model.add(Flatten(input_shape=(seq_length,))) 

	model.add(Dense(hidden_size,activation='relu'))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')
	history = model.fit(X, Y, epochs=epochs, validation_split=0.2, verbose=1, batch_size=batch_size)

	return(model)
# Train LSTM
def train_lstm(X, Y, hidden_size, seq_length, numb_features, epochs, batch_size):

	model = Sequential()
	model.add(LSTM(hidden_size, activation='relu', input_shape=(seq_length, numb_features)))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')
	history = model.fit(X, Y, epochs=epochs, validation_split=0.2, verbose=1, batch_size=batch_size)


	return(model)

# Train BILSTM
def train_bilstm(X, Y, hidden_size, seq_length, numb_features, epochs, batch_size):
	
	model = Sequential()
	model.add(Bidirectional(LSTM(hidden_size, activation='relu', input_shape=(seq_length, numb_features))))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')
	history = model.fit(X, Y, epochs=epochs, validation_split=0.2, verbose=1, batch_size=batch_size)

	return(model)


# Train RNN
def train_rnn(X, Y, hidden_size, seq_length, numb_features, epochs, batch_size):

	model = Sequential()
	model.add(SimpleRNN(hidden_size, activation='relu', input_shape=(seq_length, numb_features)))
	model.add(Dense(1))
	model.compile(optimizer='adam', loss='mse')
	history = model.fit(X, Y, epochs=epochs, validation_split=0.2, verbose=1, batch_size=batch_size)


	return(model)



# Load model and training parameters
def param():
	parser = argparse.ArgumentParser()

	parser.add_argument("--model", help="lstm or bilstm or rnn", type=str, default="lstm")
	parser.add_argument("--train_size", help="", type=int, default=100)
	parser.add_argument("--seq_length", help="", type=int, default=2)
	parser.add_argument("--max_int", help="", type=int, default=10)
	
	parser.add_argument("--hidden_size", help="", type=int, default=100)
	parser.add_argument("--numb_features", help="", type=int, default=1)
	parser.add_argument("--batch_size", help="", type=int, default=4)
	parser.add_argument("--epochs", help="", type=int, default=100)
	parser.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True)
		
	args = parser.parse_args()
	return(args)

#-----------------------------------------
# Main
#-----------------------------------------

if __name__=='__main__':


	args = param()
	

	train_size = args.train_size
	seq_length = args.seq_length
	max_int = args.max_int
	hidden_size = args.hidden_size
	numb_features = args.numb_features
	batch_size = args.batch_size
	epochs = args.epochs


	# Generate training data
	(X, Y) = gen_train_data(train_size, seq_length, max_int)
	
	
	# Train -------------------------------- 
	if args.model == "fnn":	
		m = train_fnn(X, Y, hidden_size, seq_length, numb_features, epochs, batch_size)

	if args.model == "rnn":	
		m = train_rnn(X, Y, hidden_size, seq_length, numb_features, epochs, batch_size)

	if args.model == "lstm":

		m = train_lstm(X, Y, hidden_size, seq_length, numb_features, epochs, batch_size)	
		
	if args.model == "bilstm":	
		m = train_bilstm(X, Y, hidden_size, seq_length, numb_features, epochs, batch_size)
	
		

	# Test -------------------------------- 	


	test_input = np.array([args.list])
	
	test_input = test_input.reshape((1, seq_length, 1)).astype(np.float64)
	
	print("Input")
	print((test_input.flatten()))
		
	test_output = m.predict(test_input, verbose=0)
	print("Predicted multiplication output")

	print((test_output.flatten()))
	
