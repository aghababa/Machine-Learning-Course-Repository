#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
def Majority_Error(S):
    c =  0
    for i in range(0,len(S)):
        if S[i]== 1:
            c = c+1
    c_min= min(c, len(S)-c)
    ME= c_min/len(S)
    return ME
def Expected_ME(S,A): # A= [[...], [...], ..., [...]]= partitioned S
    E_EM =  0
    for i in range (0, len(A)):
        E_EM= E_EM + ((len(A[i])/len(S))* Majority_Error(A[i]))
    return E_EM 


# In[30]:


def Informathion_Gain(S,A):
    IG = Majority_Error(S) - Expected_ME(S,A)
    return IG


# In[38]:


S=[-1,-1,1,1,1,-1,1,-1,1,1,1,1,1,-1]
A=[[-1,-1,-1,1,1], [1,1,1,1], [1,1,1,-1,-1]]


# In[40]:


print("Majority Error = ", Majority_Error(S))
print("Expected Majority Error(S,A) = ", Expected_ME(S,A))
print("Informathion Gain = ", Informathion_Gain(S,A))


# In[ ]:




