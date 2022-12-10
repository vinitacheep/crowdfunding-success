#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import nltk
import datetime
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


# In[15]:


np.random.seed(100)


# In[16]:


columns = ["location_stateprefix", "created_date", "raised_amnt", "goal_amnt", "story"]
df = pd.read_csv('disease_filtered_geotagged_surgery_campaigns.csv', usecols=columns, encoding='utf8')


# In[4]:


labels = []
story_w_feats = []
# For each of the samples...
for index, row in df.iterrows():

    # Piece it together...    
    combined = ""
    
#     #combined += "The ID of this item is {:}, ".format(row["Clothing ID"])
#     combined += "This item comes from the {:} department and {:} division, " \
#                 "and is classified under {:}. ".format(row["Department Name"], 
#                                                        row["Division Name"], 
#                                                        row["Class Name"])
    
    combined += "i am from {:}. ".format(row["location_stateprefix"])
    
   # d1 = datetime.datetime.strptime(row["created_date"], "%d.%m.%y") '2014-05-16T11:41:31-05:00'
    new_date = str(row["created_date"]).replace('T', ' ')
    # Slice string to remove last 3 characters from string
    new_date = new_date[:len(new_date) - 6]
    new_date += ".000000"
    d1 = datetime.datetime.strptime(new_date, "%Y-%m-%d %H:%M:%S.%f")
    d2 = datetime.datetime.today()
   # Then, compute their difference:

    difference = d2 - d1
    #And divide it by one day:
    difference_in_days = difference / datetime.timedelta(days=1)
    combined += "the campaign was created " + str(difference_in_days) + " days ago. "
    
    combined+= "this description is " + str(len(row["story"].split())) + " words long. "
    
    # Finally, append the review the text!
    combined += row["story"]
    
    # Add the combined text to the list.
    story_w_feats.append(combined)

    # Also record the sample's label.
    if row["raised_amnt"] >= row["goal_amnt"]:
        labels.append(1)
    else:
        labels.append(0)


# In[5]:


df_labels = pd.DataFrame({'label':labels})
df_stories = pd.DataFrame({'story':story_w_feats})
print(df_stories['story'][0])


# In[6]:



# Step - a : Remove blank rows if any.
df_stories['story'].dropna(inplace=True)
# Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
# df_stories['story'] = [entry.lower() for entry in df_stories['story']]
# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words

#print(df_stories['story'][0])
for index,entry in enumerate(df_stories['story']):
    token_list = []
    if len(entry.split()) > 250:       
        token_list.append(entry.split()[:200])
    else:
        token_list.append(word_tokenize(entry))
    df_stories.loc[index,'text_final'] = str(token_list)

    
#df_stories['story']= [word_tokenize(entry) for entry in df_stories['story']]
# Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
# tag_map = defaultdict(lambda : wn.NOUN)
# tag_map['J'] = wn.ADJ
# tag_map['V'] = wn.VERB
# tag_map['R'] = wn.ADV

# for index,entry in enumerate(df_stories['story']):
#     # Declaring Empty List to store the words that follow the rules for this step
#     Final_words = []
#     # Initializing WordNetLemmatizer()
#     word_Lemmatized = WordNetLemmatizer()
#     # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
#     for word, tag in pos_tag(entry):
#         # Below condition is to check for Stop words and consider only alphabets
#         if word not in stopwords.words('english') and word.isalpha():
#             word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
#             Final_words.append(word_Final)
#     # The final processed set of words for each iteration will be stored in 'text_final'
#     df_stories.loc[index,'text_final'] = str(Final_words)


# In[7]:


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df_stories['text_final'],df_labels['label'],test_size=0.2)


# In[8]:


Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


# In[9]:


Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(df_stories['text_final'])
Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# In[10]:


#print(Tfidf_vect.vocabulary_)


# In[17]:


def metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return acc, f1, precision, recall


# In[18]:


# fit the training dataset on the NB classifier
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)
print(metric(predictions_NB, Test_Y))


# In[ ]:


# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print(metric(predictions_SVM, Test_Y))

