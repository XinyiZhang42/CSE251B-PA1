#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Basic setups 
import numpy as np
from skimage import io
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


from dataloader import load_data
from PCA import PCA


# In[3]:


from random import shuffle
import math
def dataSplit(K,data):
    """This function is used to implement k-fold cross-validation"""
    #Shuffle the dataset to get more accurate performance
    
    #print("original shape:",data.shape)
    
    index = np.arange(0,len(data))
    shuffle(index)
    shuffled_data= data[index]
    
    
    #print("shuffled shape:",shuffled_data.shape)
    
    #Initialize list for train,test and val 
    train = []
    val = []
    test = []
    
    #Split the data into training, testing and handout set
    size = len(data)
    set_size = math.floor(size/K)

    for i in range(K):
        # select subsets of data
        test_i = shuffled_data[i*set_size:(i+1)*set_size] #ith cut as training set for ith fold
        
        #print("shape of test_i",test_i.shape)
        
        temp = (i+1)%K
        val_i = shuffled_data[temp*set_size:(temp+1)*set_size] # (i+1)%kth cut as val set for ith fold
        
        #print("shape of val_i",val_i.shape)
        
        if i < temp:
            temp1 = shuffled_data[:i*set_size]
            temp2 = shuffled_data[(temp+1)*set_size:]
            train_i = np.concatenate((temp1, temp2),axis=0)
            
        if i > temp:
            temp1 = shuffled_data[0:temp*set_size]
            temp2 = shuffled_data[(temp+1)*set_size:i*set_size]
            temp3 = shuffled_data[(i+1)*set_size:]
            train_i_temp = np.concatenate((temp1, temp2),axis=0)
            train_i = np.concatenate((train_i_temp, temp3),axis=0)
            
        #print("shape of train_i",train_i.shape)
        
        train.append(train_i)
        val.append(val_i)
        test.append(test_i)
    return train, val, test


# In[4]:


def img_flatten(img):
    """This function is used to convert 2-D images to 1-D vectors"""
    flatten_img = []
    for i in range(len(img)):
        flatten_img.append(img[i].flatten())
    return np.array(flatten_img)

def projectPC(x,mean_image, eigen_values, eigen_vectors):
    """This function is used to project the data x on the given training set x_train"""
    msd = x - mean_image
    projected_image = np.matmul(msd,eigen_vectors)/eigen_values
    projected = np.insert(projected_image,0,1,axis=1)
    return projected


# In[5]:


def one_hot(x):
    onehot = np.zeros((x.size, 4))
    onehot[np.arange(x.size),x.astype(int)[:,0]] = 1
    return onehot

def softmax(x):
    return (np.exp(x.T) / np.sum(np.exp(x), axis=1)).T

def to_class(x):
    return x.argmax(axis=1)

def accuracy(x,y,weight):
    #x = np.asmatrix(x)
    #y = np.asmatrix(y)
    #weight = np.asmatrix(weight)
    
    y_hat = softmax(np.matmul(x,weight))
    #print("shape of y_hat",y_hat.shape)
    
    prediction = to_class(y_hat)
    onehot = np.zeros((prediction.size, 4))
    onehot[np.arange(prediction.size),prediction] = 1
    prediction = onehot
    #print(prediction)
    #print(np.dot(prediction[1],y[1]))
    correct = np.zeros(y.shape[0])
    for i in range(y.shape[0]):
        if np.dot(prediction[i],y[i]) == 1:
            correct[i]=1
    accuracy = sum(correct)/len(correct) 
    return accuracy
    
def cross_entropy(x,y,weight):
    #x = np.asmatrix(x)
    #y = np.asmatrix(y)
    #weight = np.asmatrix(weight)
    #print(np.matmul(weight.T,x).shape)
    
    y_hat = softmax(np.matmul(x,weight))
    #print("shape of y,y_hat",y.shape,y_hat.shape)
    
    cost = np.multiply(-y,np.log(y_hat))
    #print("shape of cost",cost.shape)
    
    error = np.sum(cost)/(len(x))
    return error

def gradientDescent(x,y,weight,learning_rate):
    #x = np.asmatrix(x)
    #y = np.asmatrix(y)
    #weight = np.asmatrix(weight)

    #print("checkgradD,weightshape:",weight.shape)
    y_hat = softmax(np.matmul(x,weight))
    error = y_hat-y
    #print(y.shape)
    #print(error[:,0].shape)
    
    gradient = np.matmul(x.T, error) / len(x)
    gradient = np.squeeze(np.asarray(gradient))
    
    #print("checkgradD,gradientshape:",gradient[0].shape)
    
    weight_updated = weight - learning_rate*gradient
    weight_updated = np.asarray(weight_updated)
    
    #print("checkgradD,weightshapeupdate:",weight_updated.shape)
    
    return gradient,weight_updated

def SGD(x,y,weight,learning_rate):
    #x = np.asmatrix(x)
    #y = np.asmatrix(y)
    #weight = np.asmatrix(weight)

    #print("checkgradD,weightshape:",weight.shape)
    y_hat = softmax(np.multiply(x,weight.T).T)
    #print(y_hat.shape)
    error = y_hat-y
    #print(y.shape)
    #print(error[:,0].shape)
    
    gradient = np.multiply(x, error.T) / len(x)
    gradient = np.squeeze(np.asarray(gradient))
    
    #print("checkgradD,gradientshape:",gradient[0].shape)
    
    weight_updated = weight - learning_rate*gradient.T
    weight_updated = np.asarray(weight_updated)
    
    #print("checkgradD,weightshapeupdate:",weight_updated.shape)
    
    return gradient,weight_updated


# In[6]:


#Load data 
data,cnt = load_data("./aligned/")


minivan= data.get('Minivan')
convertible = data.get('Convertible')
pickup= data.get('Pickup')
sedan = data.get('Sedan')


minivan_flatten = img_flatten(minivan)
convertible_flatten = img_flatten(convertible)
pickup_flatten = img_flatten(pickup)
sedan_flatten = img_flatten(sedan)

#print("shape of minivan:",minivan_flatten.shape)
#print("shape of convertible:",convertible_flatten.shape)


# In[7]:


num_fold = 10
max_iter = 800
learning_rate = 5
num_PC = 200

trainM, valM, testM = dataSplit(num_fold,minivan_flatten)
trainC, valC, testC = dataSplit(num_fold,convertible_flatten)
trainP, valP, testP = dataSplit(num_fold,pickup_flatten)
trainS, valS, testS = dataSplit(num_fold,sedan_flatten)


# In[8]:


#Initialize accuracy and error matrix
train_error = np.zeros((max_iter,num_fold))
val_error = np.zeros((max_iter,num_fold))
test_error = np.zeros((max_iter,num_fold))
            
train_acc = np.zeros((max_iter,num_fold))
val_acc = np.zeros((max_iter,num_fold))
test_acc = np.zeros((max_iter,num_fold))

average_train = np.zeros((1,max_iter))
average_val = np.zeros((1,max_iter))

test_accuracy = np.zeros((1,num_fold))

print("shape of matrix:",train_acc.shape)

    


# In[9]:


def plot_top_PCs(eigenvectors,n):
    """This function is used plot top PCs"""
    imgs = []
    plt.figure()
    for i in range(n):
        eigen = eigenvectors[:,i]
        image = np.reshape(eigen,(200,300))
        imgs.append(image)
        plt.subplot(2,n/2,i+1)
        plt.imshow(imgs[i])


# In[10]:


for fold in range(num_fold):
    print("%dth iteration :" %(fold))
    
    #generate train,test and val set
    train = np.concatenate((trainM[fold], trainC[fold], trainP[fold], trainS[fold]),axis=0)
    test = np.concatenate((testM[fold], testC[fold], testP[fold], testS[fold]),axis=0)
    val = np.concatenate((valM[fold], valC[fold], valP[fold], valS[fold]),axis=0)

    #print("shape of train set:",train.shape)
    #print("shape of test set:",test.shape)
    #print("shape of val set:",val.shape)
    
    y_train = np.concatenate((np.zeros(len(trainM[fold])), np.ones(len(trainC[fold])), 2*np.ones(len(trainP[fold])), 3*np.ones(len(trainS[fold]))), axis=0)
    y_train= np.array([[i] for i in y_train])
    y_train = one_hot(y_train)
    #print(y_train)

    y_test = np.concatenate((np.zeros(len(testM[fold])), np.ones(len(testC[fold])), 2*np.ones(len(testP[fold])), 3*np.ones(len(testS[fold]))), axis=0)
    y_test = np.array([[i] for i in y_test])
    y_test = one_hot(y_test)

    y_val = np.concatenate((np.zeros(len(valM[fold])), np.ones(len(valC[fold])), 2*np.ones(len(valP[fold])), 3*np.ones(len(valS[fold]))), axis=0)
    y_val = np.array([[i] for i in y_val])
    y_val = one_hot(y_val)
    
    #Perform PCA to find top PCs on training set
    projected, mean_image, top_sqrt_eigen_values, top_eigen_vectors = PCA(train, num_PC)
    x_train = np.insert(projected, 0, 1, axis=1)

    #Project test and val set on top PCs
    x_test = projectPC(test,mean_image, top_sqrt_eigen_values, top_eigen_vectors)
    x_val = projectPC(val,mean_image, top_sqrt_eigen_values, top_eigen_vectors)
    
    weight = np.zeros((len(x_train[0]),4))
    #print(weight[:,0].shape)
                      
    for j in range(max_iter):
        #grad,weight = gradientDescent(x_train,y_train,weight,learning_rate)
        rand = np.arange(len(x_train))
        np.random.shuffle(rand)
        for i in range(len(x_train)):
            grad,weight = SGD(x_train[rand[i]],y_train[rand[i]],weight,learning_rate)
        
        #print("shape of weight",weight.shape)
    
        train_error[j][fold] = cross_entropy(x_train,y_train,weight)
        train_acc[j][fold] = accuracy(x_train,y_train,weight)
                                                 
        #Calculate the error for hold out set using updated weight
        val_error[j][fold] = cross_entropy(x_val,y_val,weight)
        val_acc[j][fold] = accuracy(x_val,y_val,weight)
                                                 
        #Calculate the error for test set using updated weight
        test_error[j][fold] = cross_entropy(x_test,y_test,weight)
        test_acc[j][fold] = accuracy(x_test,y_test,weight)
        
    val_temp = val_error[:,fold]
    
    #print("shape of val_temp:",val_temp.shape)
    
    index_min = np.argmin(val_temp)
    test_accuracy[0][fold] = test_acc[index_min][fold]
    

    #plot_top_PCs(top_eigen_vectors,4)
    #plt.savefig('./plots/5c_top4eign'+str(fold)+'th.png')


# In[11]:


print(train_error)


# In[12]:


#Plot average loss curves for training and val sets
std_train = np.zeros((1,max_iter))
std_val = np.zeros((1,max_iter))

for q in range(max_iter):
    average_train[0][q] = np.mean(train_error[q][:])
    average_val[0][q] = np.mean(val_error[q][:])
    std_train[0][q] = np.std(train_error[q][:])
    std_val[0][q] = np.std(val_error[q][:])
    
plt.plot(average_train[0,:],color = 'red',label = 'Training loss')


plt.title("Loss for training set with Learning rate ="+str(learning_rate)+", Number of PCs = "+str(num_PC))
plt.ylabel('Loss')
plt.xlabel('Iterations')
plt.legend()

errorbar_x = np.zeros((1,int(max_iter/50) ))
errorbar_train = np.zeros((1,int(max_iter/50)))
errorbar_val = np.zeros((1,int(max_iter/50) ))
errorbar_y_train = np.zeros((1,int(max_iter/50)))
errorbar_y_val = np.zeros((1,int(max_iter/50)))

for p in range(int(max_iter/50)):
    
    errorbar_x[0][p] = 50*(p+1)
    errorbar_train[0][p] = std_train[0][50*(p+1)-1]
    errorbar_val[0][p] = std_val[0][50*(p+1)-1]
    errorbar_y_train[0][p] = average_train[0][50*(p+1)-1]
    errorbar_y_val[0][p] = average_val[0][50*(p+1)-1]
    



plt.errorbar(errorbar_x[0,:], errorbar_y_train[0,:], errorbar_train[0,:],fmt = '.r', capsize=5)


#plt.savefig('./plots/5c_loss_train_lr_'+str(learning_rate)+'_pc_'+str(num_PC)+'.png')
plt.show()


# In[13]:


plt.plot(average_val[0,:],color = 'blue',label = 'Val loss')
plt.title("Loss for val set with Learning rate ="+str(learning_rate)+", Number of PCs = "+str(num_PC))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.errorbar(errorbar_x[0,:], errorbar_y_val[0,:], errorbar_val[0,:],fmt = '.b', capsize=5)

plt.legend()
#plt.savefig('./plots/5c_loss_val_lr_'+str(learning_rate)+'_pc_'+str(num_PC)+'.png')
plt.show()


# In[14]:


#Print testing accuracy
test_acc_final = np.mean(test_accuracy)
print("The final accuracy is :",test_acc_final)


# In[15]:


#visualize weights
weighted_PCs = []
for i in range(4):
    weighted_PCs.append(np.matmul(top_eigen_vectors[:,0:100],weight[1:101,i]).reshape((200,300)))
plt.figure(figsize=(10, 7))
plt.subplot(2,2,1)
plt.imshow(weighted_PCs[0])
plt.title('Minivan')
plt.subplot(2,2,2)
plt.imshow(weighted_PCs[1])
plt.title('Convertible')
plt.subplot(2,2,3)
plt.imshow(weighted_PCs[2])
plt.title('Pickup')
plt.subplot(2,2,4)
plt.imshow(weighted_PCs[3])
plt.title('Sedan')
plt.suptitle('Visualized weights')


# In[ ]:




