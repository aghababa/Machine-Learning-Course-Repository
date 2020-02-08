#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
def llog(p):
    if p == 0:
        return 0
    else:
        return np.log2(p)


# In[2]:


def Entropy(S,W):
    c = 0 # positive counter
    d = 0 # negative counter
    for i in range(0,len(S)):
        if S[i] > 0:
            c = c + W[i]
        else:
            d = d + W[i]
    E = - (c/(c+d)) * (llog(c/(c+d))) - (d/(c+d)) * llog(d/(c+d))
    return E


# In[3]:


def Expected_Entropy(S,A,B): # A= [[...], [...], ..., [...]]= partitioned S
    E_E = 0            # B= [[...], [...], ..., [...]]= partitioned W
    for i in range (0, len(A)):
        E_E= E_E + ((sum(B[i])/sum(W)) * Entropy(A[i],B[i]))
    return E_E


# In[4]:


def Informathion_Gain(S,A):
    I_G = Entropy(S,W) - Expected_Entropy(S,A,B)
    return I_G


# In[7]:


# (Outlook= Sunny) Wind
W=[1,1,1,1,1,5/14]
S=[-1,-1,-1,1,1,1]
A=[[-1,-1,1,1],[1,-1]]
B=[[1,1,1,5/14],[1,1]]


# In[8]:


print("Entropy = ", Entropy(S,W))
print("Expected Entropy(S,A) = ", Expected_Entropy(S,A,B))
print("Informathion Gain = ", Informathion_Gain(S,A))


# In[9]:


# (Outlook= Sunny) Humidity
W=[1,1,1,1,1,5/14](Outlook= Sunny) 
S=[-1,-1,-1,1,1,1]
A=[[-1,-1,-1],[1,1,1]]
B=[[1,1,1],[1,1,5/14]]


# In[10]:


print("Entropy = ", Entropy(S,W))
print("Expected Entropy(S,A) = ", Expected_Entropy(S,A,B))
print("Informathion Gain = ", Informathion_Gain(S,A))


# In[11]:


# (Outlook= Sunny) Temperature 
W=[1,1,1,1,1,5/14]
S=[-1,-1,-1,1,1,1]
A=[[-1,-1],[-1,1,1],[1]]
B=[[1,1],[1,1,5/14],[1]]


# In[12]:


print("Entropy = ", Entropy(S,W))
print("Expected Entropy(S,A) = ", Expected_Entropy(S,A,B))
print("Informathion Gain = ", Informathion_Gain(S,A))


# In[13]:


# (Outlook= Rain) Wind
W=[1,1,1,1,1,5/14]
S=[-1,-1,1,1,1,1]
A=[[1,1,1,1],[-1,-1]]
B=[[1,1,1,5/14],[1,1]]


# In[14]:


print("Entropy = ", Entropy(S,W))
print("Expected Entropy(S,A) = ", Expected_Entropy(S,A,B))
print("Informathion Gain = ", Informathion_Gain(S,A))


# In[15]:


# (Outlook= Rain) Humidity
W=[1,1,1,1,1,5/14]
S=[-1,-1,1,1,1,1]
A=[[-1,1,1,1],[1,-1]]
B=[[1,1,1,5/14],[1,1]]


# In[16]:


print("Entropy = ", Entropy(S,W))
print("Expected Entropy(S,A) = ", Expected_Entropy(S,A,B))
print("Informathion Gain = ", Informathion_Gain(S,A))


# In[17]:


# (Outlook= Rain) Temperature 
W=[1,1,1,1,1,5/14]
S=[-1,-1,1,1,1,1]
A=[[-1,1,1,1],[1,-1]]
B=[[1,1,1,5/14],[1,1]]


# In[18]:


print("Entropy = ", Entropy(S,W))
print("Expected Entropy(S,A) = ", Expected_Entropy(S,A,B))
print("Informathion Gain = ", Informathion_Gain(S,A))


# In[ ]:




