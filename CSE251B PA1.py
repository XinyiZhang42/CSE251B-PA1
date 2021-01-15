#!/usr/bin/env python
# coding: utf-8

# ## CSE251B PA1

# In[1]:


#Basic setups 
import numpy as np
from skimage import io
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ### Part 1: Load and preprocess the data

# In[7]:


from dataloader import load_data
from PCA import PCA


# In[9]:


#Load and preprocess the data
#We change the load_data function by deleting the value assigned to datatype in the definition
aligned_data,cnt = load_data("./aligned/")


# ### Part 2: Cross Validation Procedure

# In[38]:


from random import shuffle
import math
def kFold(K,data):
    """This function is used to implement k-fold cross-validation"""
    #Shuffle the dataset to get more accurate performance
    index = np.arange(0,len(data))
    shuffled_idx = np.random.shuffle(index)
    shuffled_data = data[shuffled_idx]
    
    #Split the data into training, testing and handout set
    size = len(data)
    set_size = math.floor(size/K)
    index_train = int(set_size*(K-2))
    index_test = int(set_size*(K-1))

    
    training_set = shuffled_data[:index_train]
    testing_set = shuffled_data[index_train:index_test]
    val_set = shuffled_data[index_test:]
    return training_set,testing_set,val_set


# In[32]:


def img_flatten(img):
    """This function is used to convert 2-D images to 1-D vectors"""
    flatten_img = []
    for i in range(len(img)):
        flatten_img.append(img[i].flatten())
    return np.array(flatten_img)


# In[29]:


def plot_top_PCs(eigenvectors,n):
    imgs = []
    plt.figure()
    for i in range(n):
        eigen = rigenvectors[i]
        image = np.reshape(eigen,(200,300))
        image = Image.fromarray(image)
        imgs.append(img)
        plt.subplot(1,n,i)
        plt.imshow(imgs[i])


# In[41]:


#Train Process
#Load data 
minivan = aligned_data.get('Minivan')
convertible = aligned_data.get('Convertible')

minivan_flatten = img_flatten(minivan)
convertible_flatten = img_flatten(convertible)

trainM,testM,valM = kFold(10,minivan_flatten)
trainC,testC,valC = kFold(10,convertible_flatten)

#Perform PCA to find top PCs
num_Pc = 3
#projected, mean_image, top_sqrt_eigen_values, top_eigen_vectors = PCA(traindata,num_PC)


# In[ ]:




