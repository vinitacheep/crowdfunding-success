#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#import dataframe
df = pd.read_csv("/Users/advait/Desktop/Forrester Group/Medical Crowdfunding/geotagged_surgery_campaigns.csv")

#filter campaigns w/ more stringent surgery criteria
df['story'] = df['story'].str.lower()

df = df[(df['story'].str.contains('surgery', na = False)) |
       (df['story'].str.contains('surgeries', na = False)) |
       (df['story'].str.contains('surgical', na = False))]

#remove 2021 data
df = df[df.year != float(2021)]


# In[ ]:


#inspect first 50 campaign stories
story_list = df['story'].head(50)

for item in story_list:
    print(item)
    print("\n ")


# In[ ]:


df.columns


# In[ ]:


#save dataset for plotting year
amount_df = df[['year', 'raised_amnt']].copy()
amount_df = amount_df.groupby("year").sum()
amount_df.to_csv("/Users/advait/Desktop/Forrester Group/Medical Crowdfunding/Figure Datasets and Plotting/total_amounts.csv")


# In[ ]:


#get counts
num_campaigns = df['year'].value_counts()


# In[ ]:


num_campaigns.to_csv("/Users/advait/Desktop/Forrester Group/Medical Crowdfunding/Figure Datasets and Plotting/num_campaigns.csv")


# In[ ]:


num_campaigns


# In[ ]:




