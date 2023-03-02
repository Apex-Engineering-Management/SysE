#!/usr/bin/env python
# coding: utf-8

# In[1]:


cost = 100000
salvage = 10000
life = 5
depreciation = (cost - salvage) / life
print("The straight-line depreciation for the machine is ${} per year.".format(depreciation))

