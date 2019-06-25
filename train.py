import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from math import sqrt

class tfMLPRegressor():
    '''
    Class Implementing a MLP in TF for time series prediction.
    '''
    def __init__(self, X, y):
        '''
        Initializes and prepares the data.
        
        :param X:       data, regressor(s), independent variable(s)
        :type  X:       numpy array with shape [number of samples, number of features]
                        Notice that the number of features, in a time series problem
                        with no exogenous input, corresponds to the past steps we want
                        to use to predict the next step(s).
        
        :param y:       response(s), dependent variable(s)
        :type  y:       numpy array with shape [number of samples, number of responses]
                        Notice that the number of responses, in a time series problem, 
                        corresponds to the number of steps that we want to predict.
        '''
        # divide X and y into training and testing sets. shuffle = False since
        # with time series data there is time correlation. If this class is to be used for 
        # standard regression, shuffle  = True.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, shuffle = False, test_size = 0.4)
        
    def declareTFVariables(self, nh1, alpha, epochs, batch_size):
        '''
        Starts declaring the various placeholders for the computational graph. In this example
        the MLP only has one layer.
        
        :param        nh1:     number of nodes in hidden layer 1
        :type         nh1:     int
        
        :param      alpha:     learning rate for the optimizer
        :type       alpha:     float, possibly smaller than 1. Ideally 0.0005
	
	:param     epochs:     number of epochs. Each epoch is a iteration of the algorithm across
			       the whole training set.
	:type      epochs:     int
	
	:param batch_size:     number of training examples in one batch.
	:type  batch_size:     int
	
        '''
        # X and y, where None will be replaced with batch_size
        x = tf.placeholder(tf.float32, [None, self.X_train.shape[1]], name = 'x')  # number of features == number of columns
        y = tf.placeholder(tf.float32, [None, self.y_train.shape[1]], name = 'y')  # number of outputs  == number of columns
        
        # weights and biases from the input to first hidden layer
        W1 = tf.Variable(tf.truncated_normal([self.X_train.shape[1], nh1] , stddev = 0.03), name = 'W1')  # n_cols(X) == n_rows(W1) for XW1 + b1
        b1 = tf.Variable(tf.truncated_normal([nh1], stddev = 0.03, name = 'b1')
        
        # weights and biases from the first hidden layer to the output layer
        W2 = tf.Variable(tf.truncated_normal([nh1, self.y_train.shape[1]] , stddev = 0.03), name = 'W2')  # n_cols(XW1+b1) == n_rows(W2)
        b2 = tf.Variable(tf.truncated_normal([self.y_train.shape[1]], stddev = 0.03, name = 'b2')
        
        # declare the outputs
        x_norm = tf.nn.l2_normalize(x)
        W1_norm = tf.nn.l2_normalize(W1)
        hidden_1_out = tf.relu(tf.add(tf.matmul(x_norm, W1_norm), b1))
        
        hidden_1_out_norm = tf.nn.l2_normalize(hidden_1_out)
        W2_norm = tf.nn.l2_normalize(W2)
        y_ = tf.relu(tf.add(tf.matmul(hidden_1_out_norm, W2_norm), b2))   # final output of the network
        
        # define a loss function
        mse = tf.losses.mean_squared_error(labels = y, predictions = y_)   # simple mean squared error loss function
        
        # create an optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate = alpha).minimize(mse)   #BGD with Adam efficient version
        
        # initiate the variables
        init_op = tf.global_variables_initializer()
        
        # now run
        with tf.Session() as sess:
		# initialize variables
		sess.run(init_op)
		# find the total number of batches
		total_batch = int(len(self.X_train) / batch_size)
		# now run for each epoch
		for epoch in range(epochs):
			avg_cost = 0
			for i in range(total_batch):
				start = i * batch_size
				end   = min(i*batch_size + batch_size, len(self.X_train))
				batch_x, batch_y = self.X_train[start:end, :], self.y_train[start:end, :]
				_, c = sess.run([optimizer, mse], feed_dict = {x: x_batch, y: y_batch})
				avg_cost += c / total_batch
			if epoch % 100 ==0:
				print('Epoch:', (epoch +1), 'cost =', '{:.3f}.format(avg_cost))
		print(sqrt(sess.run(mse, feed_dict = {x: self.X_test, y: self.y_test})))
		# keep the prediction for later
		pred = sess.run(y_, feed_dict = {x:X_test})
		     
		
            
        
		
