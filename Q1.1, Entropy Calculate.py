#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def Entropy(p):
    if p == 1:
        return 0
    elif p == 0:
        return 0
    else:
        E = - p * np.log2(p)- (1-p) * np.log2(1-p)
        return E

def Expected_Entropy(p,a,b): 
    E_E = p * Entropy(a) + (1-p) * Entropy(b)
    return E_E 


# In[2]:


def Informathion_Gain(S,A):
    IG = Entropy(S) - Expected_Entropy(p,F1,F2)
    return IG


# In[3]:


p= 1/3 # proportion of examples for this attribute
F1= 1 # fraction of 0's in child_1 attribute
F2= 0 # fraction of 1's in child_2 attribute
S = 1/3 # fraction of 0's or 1's for the father attribute
x= Entropy(F1)
y= Entropy(F2)
A= Expected_Entropy(p,F1,F2)


# In[4]:


print("Entropy1 = ", x)
print("Entropy2 = ", y)
print("Expected_Entropy = ", Expected_Entropy(p,F1,F2))
print("Information Gain = ", Informathion_Gain(S,A))


# In[ ]:




