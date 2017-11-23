def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

import numpy as np
cat_data = np.zeros((10000,3073))
train_data = unpickle('data_batch_1')
for i in range(0,10000):
	 if train_data['labels'][i] == 3:
		 cat_data[i,0:3072] = train_data['data'][i]
		 cat_data[i,3072] = 1
	 else:
		 cat_data[i,0:3072] = train_data['data'][i]
		 cat_data[i,3072] = 0
cat_data_train = np.zeros((600,3073))
cat_data_dev = np.zeros((200,3073))
cat_data_set = np.zeros((200,3073))
cat_data_train = cat_data[0:600,:]

cat_data_dev = cat_data[600:800,:]
cat_data_set = cat_data[800:1000,:]
cat_data_train[:,0:3072] = cat_data_train[:,0:3072]  - np.mean(cat_data_train[:,0:3072],axis = 0)
cat_data_dev[:,0:3072] = cat_data_dev[:,0:3072]  - np.mean(cat_data_dev[:,0:3072],axis = 0)
cat_data_train[:,0:3072] = cat_data_train[:,0:3072]  / np.std(cat_data_train[:,0:3072],axis = 0)
cat_data_dev[:,0:3072] = cat_data_dev[:,0:3072]  / np.std(cat_data_dev[:,0:3072],axis = 0)

def sigmoid(z):

	y = 1/(1+np.exp(-z))
	
	return y
	
def initializer(dimm):
	w = np.zeros((dimm,1))
	b = 0 
	
	return w,b

def  propagate(w,b,x,y):
	
	m = x.shape[1]
	temp = np.dot(np.transpose(w),x) + b 
	A = sigmoid(temp)
	
	loss_function = np.sum( y * np.log(A) + (1-y) * np.log(1-A)) *(-1)/m
	
	db = np.sum(A - y)/m
	dw = (np.dot(x,np.transpose(A - y)))/m
	
	
	return dw, db, loss_function
#w, b, x, y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])

 


def optimization ( w,b,x,y,num_iteration,learn_rate):
	
	costs = [] 
	for i in range(num_iteration):
		
		dw,db,cost = propagate(w,b,x,y)
		w = w - learn_rate*dw 
		b = b - learn_rate*db
	
		
		if i%10 == 0:
			costs.append(cost)
		#print cost
			
	return w,b,costs
	

#print optimization(w,b,x,y,100,0.009)

def predict(w,b,x):
	
	
	A = sigmoid(np.dot(np.transpose(w),x) + b) 
	
	m = A.shape[1] 
	A = A>0.5
			
	return A
		

		

		
		
def logistic_modelef(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate):
	X_train = np.transpose(X_train)
	Y_train = np.transpose(Y_train)
	X_test = np.transpose(X_test)
	Y_test = np.transpose(Y_test)
	m = X_train[0]
	w,b = initializer(X_train.shape[0])
	w_l,b_l,loss = optimization(w,b,X_train,Y_train,num_iterations,learning_rate) 
	output = predict(w_l,b_l,X_test) 
	temp = np.dot(np.transpose(w_l),X_test) + b_l 
	B = sigmoid(temp)
	sum_loss= np.sum( Y_test * np.log(B) + (1-Y_test) * np.log(1-B)) *(-1)/m
	
	Y_test = np.reshape(Y_test,(1,200))
	Y_test = np.array(Y_test)
	output = np.array(output) 
	print output.shape
	print Y_test.shape
	model_error = np.sum(Y_test * (1-output)  + (1-Y_test) * (output),axis=1) /200
	return w_l,b_l,loss,sum_loss,model_error
	
a,b,c,d,o = logistic_modelef(cat_data_train[:,0:3072],cat_data_train[:,3072],cat_data_dev[:,0:3072],cat_data_dev[:,3072],2000,0.01)
import matplotlib.pyplot as plt
plt.plot(c)
plt.show()
#print cat_data_train[1,1:100]
#print cat_data_train.shape
print o


