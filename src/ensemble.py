
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[4]:


result1 = pd.read_csv("./model_1.csv")
result2 = pd.read_csv("./model_2.csv")
result3 = pd.read_csv("./model_3.csv")
result4 = pd.read_csv("./model_4.csv")
result5 = pd.read_csv("./model_5.csv")
result6 = pd.read_csv("./model_6.csv")
result7 = pd.read_csv("./model_7.csv")
result8 = pd.read_csv("./model_8.csv")
result9 = pd.read_csv("./model_9.csv")
result10 = pd.read_csv("./model_10.csv")
result11 = pd.read_csv("./model_11.csv")
result12 = pd.read_csv("./model_12.csv")


# In[5]:


result_F = []
result_F.append(result1['id'])
for i in result1.columns:
    if i!='id':
        result_F.append((result1[i]+result2[i]+result3[i]+result4[i]+result5[i]+result6[i]+result7[i]+                        result8[i]+result9[i]+result10[i]+result11[i]+result12[i])/12)


# In[6]:


ens_result = pd.DataFrame(result_F).T
ens_result.to_csv("result.csv",index=None)


# In[5]:


