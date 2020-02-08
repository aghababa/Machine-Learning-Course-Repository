#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
def llog(p):
    if p == 0:
        return 0
    else:
        return np.log2(p)


# In[7]:


def Entropy(S):
    c = 0 # positive counter
    for i in range(0,len(S)):
        if S[i] > 0:
            c = c + 1
    E = - (c/len(S)) * (llog(c/len(S))) - (1- c/len(S)) * llog(1- c/len(S))
    return E


# In[8]:


def Expected_Entropy(S,A): # A= [[...], [...], ..., [...]]= partitioned S
    E_E = 0            # B= [[...], [...], ..., [...]]= partitioned W
    for i in range (0, len(A)):
        E_E= E_E + ((sum(A[i])/sum(S)) * Entropy(A[i]))
    return E_E


# In[9]:


def Informathion_Gain(S,A):
    I_G = Entropy(S) - Expected_Entropy(S,A)
    return I_G


# In[25]:


# Wind
S=[-1,-1,1,1,1,-1,1,-1,1,1,1,1,1,-1,1]
A=[[1,1,1,1,1,1,-1,-1,1],[1,1,1,-1,-1,-1]]


# In[26]:


print("Entropy = ", Entropy(S))
print("Expected Entropy(S,A) = ", Expected_Entropy(S,A))
print("Informathion Gain = ", Informathion_Gain(S,A))


# In[27]:


# Outlook
S=[-1,-1,1,1,1,-1,1,-1,1,1,1,1,1,-1,1]
A=[[1,1,-1,-1,-1],[1,1,1,1,1],[1,1,1,-1,-1]]


# In[28]:


print("Entropy = ", Entropy(S))
print("Expected Entropy(S,A) = ", Expected_Entropy(S,A))
print("Informathion Gain = ", Informathion_Gain(S,A))


# In[29]:


# Humidity
S=[-1,-1,1,1,1,-1,1,-1,1,1,1,1,1,-1,1]
A=[[1,1,1,-1,-1,-1,-1],[1,1,1,1,1,1,-1,1]]


# In[30]:


print("Entropy = ", Entropy(S))
print("Expected Entropy(S,A) = ", Expected_Entropy(S,A))
print("Informathion Gain = ", Informathion_Gain(S,A))


# In[31]:


# Temperature 
S=[-1,-1,1,1,1,-1,1,-1,1,1,1,1,1,-1,1]
A=[[1,1,-1,-1],[1,1,1,1,-1,-1,1],[1,1,1,-1]]


# In[32]:


print("Entropy = ", Entropy(S))
print("Expected Entropy(S,A) = ", Expected_Entropy(S,A))
print("Informathion Gain = ", Informathion_Gain(S,A))


# In[ ]:





# In[ ]:




