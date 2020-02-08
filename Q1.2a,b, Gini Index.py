#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
def Gini_Index(S):
    c =  0
    for i in range(0,len(S)):
        if S[i]== 1:
            c = c+1
    GI= 1- ((c/len(S))**2 + ((len(S)-c)/len(S))**2)
    return GI
def Expected_GI(S,A): # A= [[...], [...], ..., [...]]= partitioned S
    E_GI =  0
    for i in range (0, len(A)):
        E_GI= E_GI + ((len(A[i])/len(S))* Gini_Index(A[i]))
    return E_GI 


# In[18]:


def Informathion_Gain(S,A):
    IG = Gini_Index(S) - Expected_GI(S,A)
    return IG


# In[46]:


S=[-1,-1,1,1,1]
A=[[-1,1,1], [-1,1]]


# In[47]:


print("Gini Index = ", Gini_Index(S))
print("Expected Gini Index(S,A) = ", Expected_GI(S,A))
print("Informathion Gain = ", Informathion_Gain(S,A))


# In[ ]:




