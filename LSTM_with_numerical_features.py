#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
import seaborn as sns
import logging
# import logger
from transformers import AdamW as AdamW_HF, get_linear_schedule_with_warmup
from torch import nn
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset
import sklearn
from sklearn import model_selection
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn.functional as F
from torch.autograd import Variable
# Define metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelBinarizer
import datetime
import numpy as np
import time
from time import perf_counter
import tqdm
from tqdm import tqdm
# import logger.info
# import logging.info
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, transforms
import scikitplot as skplt
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder





# In[2]:


logging.getLogger().setLevel(logging.INFO)
columns = ["location_stateprefix", "created_date", "raised_amnt", "goal_amnt", "story"]
df = pd.read_csv('disease_filtered_geotagged_surgery_campaigns.csv', usecols=columns, encoding='utf8')
print(df.head())


# In[3]:


import re

def preprocess(text):
    """
    This function takes a string as input, then performs these operations: 
        - lowercase
        - remove URLs
        - remove ticker symbols 
        - removes punctuation
        - removes any single character tokens
    Parameters
    ----------
        message : The text message to be preprocessed
    Returns
    -------
        text: The preprocessed text
    """ 
    # Replace everything not a letter or apostrophe with a space
    text = re.sub('[^a-zA-Z\']', ' ', text)
    # Remove single letter words
    text = ' '.join( [w for w in text.split() if len(w)>1] )
    
    return text
        
# Process for all messages
preprocessed = [preprocess(story) for story in df["story"]]


# In[4]:


print(preprocessed[0])


# In[5]:




def tokenize_text(text, option):
  '''
  Tokenize the input text as per specified option
    1: Use python split() function
    2: Use regex to extract alphabets plus 's and 't
    3: Use NLTK word_tokenize()
    4: Use NLTK word_tokenize(), remove stop words and apply lemmatization
  '''
  if option == 1:
    return text.split()
  elif option == 2:
    return re.findall(r'\b([a-zA-Z]+n\'t|[a-zA-Z]+\'s|[a-zA-Z]+)\b', text)
  elif option == 3:
    return [word for word in word_tokenize(text) if (word.isalpha()==1)]
  elif option == 4:
    words = [word for word in word_tokenize(text) if (word.isalpha()==1)]
    # Remove stop words
    stop = set(stopwords.words('english'))
    words = [word for word in words if (word not in stop)]
    # Lemmatize words (first noun, then verb)
    wnl = nltk.stem.WordNetLemmatizer()
    lemmatized = [wnl.lemmatize(wnl.lemmatize(word, 'n'), 'v') for word in words]
    return lemmatized
  else:
    logging.warn("Please specify option value between 1 and 4")
    return []


# In[6]:


# Create vocab
def create_vocab(messages, show_graph=False):
    corpus = []
    for message in messages:
        tokens = tokenize_text(message, 3) # Use option 3
        corpus.extend(tokens)
    #print("number of all words: " + str(len(corpus)))
    logging.info("The number of all words: {}".format(len(corpus)))

    # Create Counter
    counts = Counter(corpus)
   # print(counts)
    logging.info("The number of unique words: {}".format(len(counts)))

    # Create BoW
    bow = sorted(counts, key=counts.get, reverse=True)
    #print(bow)
    logging.info("Top 40 frequent words: {}".format(bow[:40]))

    # Indexing vocabrary, starting from 1.
    vocab = {word: ii for ii, word in enumerate(counts, 1)}
    id2vocab = {v: k for k, v in vocab.items()}

    if show_graph:
        from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
        # Generate Word Cloud image
        text = " ".join(corpus)
        stopwords = set(STOPWORDS)
        stopwords.update(["will", "report", "reporting", "market", "stock", "share"])

        wordcloud = WordCloud(stopwords=stopwords, max_font_size=50, max_words=100, background_color="white", collocations=False).generate(text)
        plt.figure(figsize=(15,7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()

        # Show most frequent words in a bar graph
        most = counts.most_common()[:80]
        x, y = [], []
        for word, count in most:
            if word not in stopwords:
                x.append(word)
                y.append(count)
        plt.figure(figsize=(12,10))
        sns.barplot(x=y, y=x)
        plt.show()

    return vocab

vocab= create_vocab(preprocessed, True)


# In[7]:


labels = []
campaign_ages = []
story_lengths = []
# creating instance of labelencoder
labelencoder = LabelEncoder()
# Assigning numerical values and storing in another column
df['state'] = labelencoder.fit_transform(df['location_stateprefix'])
states = df["state"].values.tolist()
#print(df.head())
# state_onehot = LabelBinarizer().fit_transform(df.location_stateprefix)
# states = state_onehot
#states = state_onehot.tolist()

# story_w_feats = []
# For each of the samples...
for index, row in df.iterrows():

    # Piece it together...    
#     combined = ""
    
# #     #combined += "The ID of this item is {:}, ".format(row["Clothing ID"])
# #     combined += "This item comes from the {:} department and {:} division, " \
# #                 "and is classified under {:}. ".format(row["Department Name"], 
# #                                                        row["Division Name"], 
# #                                                        row["Class Name"])
    
#     combined += "i am from {:}. ".format(row["location_stateprefix"])
    
#    # d1 = datetime.datetime.strptime(row["created_date"], "%d.%m.%y") '2014-05-16T11:41:31-05:00'
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
    campaign_ages.append(difference_in_days)
#     combined += "the campaign was created " + str(difference_in_days) + " days ago. "
    
#     combined+= "this description is " + str(len(row["story"].split())) + " words long. "
    story_lengths.append(len(row["story"].split()))
    

#     # Finally, append the review the text!
#     combined += row["story"]
    
#     # Add the combined text to the list.
#     story_w_feats.append(combined)

    # Also record the sample's label.
    if row["raised_amnt"] >= row["goal_amnt"]:
        labels.append(1)
    else:
        labels.append(0)
data_tuples = list(zip(story_lengths, states, campaign_ages))
meta_df = pd.DataFrame(data_tuples, columns=['story_length','state', 'campaign_age'])
print(meta_df.head())
print(meta_df.iloc[200,:])


# In[8]:


# for index,entry in enumerate(df_stories['story']):
#     token_list = []
#     if len(entry.split()) > 250:       
#         token_list.append(entry.split()[:250])
#     else:
#         token_list.append(word_tokenize(entry))
#     df_stories.loc[index,'text_final'] = str(token_list)


# In[9]:



# Define LSTM Model
class LstmTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_size, dense_size, numeric_feature_size, output_size, lstm_layers=2, dropout=0.1):

        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_size = lstm_size
        self.dense_size = dense_size
        self.output_size = output_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, lstm_size, lstm_layers, dropout=dropout, batch_first=False)
        self.dropout = nn.Dropout(dropout)
        # Insert an additional fully connected when combining with other inputs
        #if dense_size == 0:
        self.fc = nn.Linear(lstm_size, output_size)
       # else:
        self.fc1 = nn.Linear(lstm_size, dense_size)
        self.fc2 = nn.Linear(dense_size + numeric_feature_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size):

        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(),
                  weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())

        return hidden

    def forward(self, nn_input_text, nn_input_meta, hidden_state):

        batch_size = nn_input_text.size(0)
        nn_input_text = nn_input_text.long()
        embeds = self.embedding(nn_input_text)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        # Stack up LSTM outputs, apply dropout
        lstm_out = lstm_out[-1,:,:]
        lstm_out = self.dropout(lstm_out)
        # Insert an additional fully connected when combining with other inputs
        #if self.dense_size == 0:
            #out = self.fc(lstm_out)
        #else:
        dense_out = self.fc1(lstm_out)
       # print(nn_input_meta)
        concat_layer = torch.cat((dense_out, torch.tensor(np.array(nn_input_meta).astype('float32'))), 1)
        out = self.fc2(concat_layer)
        # Softmax
        logps = self.softmax(out)

        return logps, hidden_state
      
 #Define LSTM Tokenizer
def tokenizer_lstm(X, vocab, seq_len, padding):

    X_tmp = np.zeros((len(X), seq_len), dtype=np.int64)
    for i, text in enumerate(X):
        #tokens = tokenize_text(np.array2string(text), 3) 
        tokens = tokenize_text(text, 3)
        token_ids = [vocab[word] for word in tokens]
        end_idx = min(len(token_ids), seq_len)
        if padding == 'right':
            X_tmp[i,:end_idx] = token_ids[:end_idx]
        elif padding == 'left':
            start_idx = max(seq_len - len(token_ids), 0)
            X_tmp[i,start_idx:] = token_ids[:end_idx]

    return torch.tensor(X_tmp, dtype=torch.int64)


# In[10]:



# Use pre-trained models for BERT
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# In[11]:



# Define a DataSet Class which simply return (x, y) pair
class SimpleDataset(Dataset):
    #add df here?
    def __init__(self, text_sample, meta_sample, y_sample):
        self.df = df
        self.datalist=[(text_sample[i], meta_sample[i], y_sample[i]) for i in range(len(y_sample))]
        #self.transform = transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        return len(self.datalist)
    def __getitem__(self,idx): #self should have text, meta, label
        
        #new
#         feats = [feat for feat in self.df.columns if feat not in [self.label_name,self.id_col]]
#         feats  = np.array(self.frame[feats].iloc[idx])
#         feats = torch.from_numpy(feats.astype(np.float32))
        #end new
        return self.datalist[idx]
    
#to process tuples
def collate_fn(data):
    text_batch, meta_batch, labels = zip(*data)
    return text_batch, meta_batch, labels
      
# Data Loader
def create_data_loader(X_text, X_meta, y, indices, batch_size, shuffle):
    X_sampled = np.array(X_text, dtype=object)[indices]
    #added
    X_meta_sampled = np.array(X_meta, dtype=object)[indices]
    y_sampled = np.array(y)[indices].astype(int)
    dataset = SimpleDataset(X_sampled, X_meta_sampled, y_sampled)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader


# In[12]:




def train_cycles(X_all, X_meta, y_all, vocab, num_samples, model_type, epochs, patience, batch_size, seq_len, lr, clip, log_level):
    result = pd.DataFrame(columns=['Accuracy', 'F1(macro)', 'Total_Time', 'ms/text'], index=num_samples, dtype = object)

    for n in num_samples:
        print("")
        logging.info("############### Start training for %d samples ###############" %n)

        # Stratified sampling
        train_size = n / len(y_all)
        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, test_size=train_size*0.2 , random_state=0)
        train_indices, valid_indices = next(sss.split(X_all, y_all))

        # Sample input data
        train_loader = create_data_loader(X_all, X_meta, y_all, train_indices, batch_size, True)
        #print(train_loader.shape())
        valid_loader = create_data_loader(X_all, X_meta, y_all, valid_indices, batch_size, False)
    #changed from 5 to 2 for lstm and bert
        if model_type == 'LSTM':
            model = LstmTextClassifier(len(vocab)+1, embed_size=512, lstm_size=1024, dense_size=0, numeric_feature_size=3, output_size=2, lstm_layers=4, dropout=0.2)
            model.embedding.weight.data.uniform_(-1, 1)
        elif model_type == 'BERT':
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        start_time = time.perf_counter() # use time.process_time() for CPU time
        acc, f1, model_trained = train_nn_model(model, model_type, train_loader, valid_loader, vocab, epochs, patience, batch_size, seq_len, lr, clip, log_level)
        end_time = time.perf_counter() # use time.process_time() for CPU time
        duration = end_time - start_time
        logging.info("Process Time (sec): {}".format(duration))
        result.loc[n] = (round(acc,4), round(f1,4), duration, duration/n*1000)

    return result, model_trained


def metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1


# In[13]:



def train_nn_model(model, model_type, train_loader, valid_loader, vocab, epochs, patience, batch_size, seq_len, lr, clip, log_level):
    # Set variables
   # logger = set_logger('sa_tweet_inperf', log_level)
    num_total_opt_steps = int(len(train_loader) * epochs)
    eval_every = len(train_loader) // 5
    warm_up_proportion = 0.1
    logging.info('Total Training Steps: {} ({} batches x {} epochs)'.format(num_total_opt_steps, len(train_loader), epochs))

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW_HF(model.parameters(), lr=lr, correct_bias=False) 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_total_opt_steps*warm_up_proportion, num_training_steps=num_total_opt_steps)  # PyTorch scheduler
    criterion = nn.NLLLoss()

    # Set Train Mode
    model.train()

    # Initialise
    acc_train, f1_train, loss_train, acc_valid, f1_valid, loss_valid = [], [], [], [], [], []
    best_f1, early_stop, steps = 0, 0, 0
    class_names = ['0:Unsuccessful','1:Successful']

    for epoch in tqdm(range(epochs), desc="Epoch"):
        logging.info('================     epoch {}     ==============='.format(epoch+1))

        #################### Training ####################
        # Initialise
        loss_tmp, loss_cnt = 0, 0
        y_pred_tmp, y_truth_tmp = [], []
        hidden = model.init_hidden(batch_size) if model_type == "LSTM" else None
        for i, batch in enumerate(train_loader):
            text_batch, meta_batch, labels = batch
            # Skip the last batch of which size is not equal to batch_size
            #labels.size(0)
            if len(labels) != batch_size:
                break
            steps += 1
           
            # Reset gradient
            model.zero_grad()
            optimizer.zero_grad()

            # Initialise after the previous training
            if steps % eval_every == 1:
                y_pred_tmp, y_truth_tmp = [], []

            if model_type == "LSTM":
                # Tokenize the input and move to device
                text_batch = tokenizer_lstm(text_batch, vocab, seq_len, padding='left').transpose(1,0).to(device)
                labels = torch.tensor(labels, dtype=torch.int64).to(device)

                # Creating new variables for the hidden state to avoid backprop entire training history
                hidden = tuple([each.data for each in hidden])
                for each in hidden:
                    each.to(device)

                # Get output and hidden state from the model, calculate the loss
                #new
                logits, hidden = model(text_batch, meta_batch, hidden) #give meta data here? get ith row of metadata
                loss = criterion(logits, labels)
                
            elif model_type == 'BERT':
                # Tokenize the input and move to device
                # Tokenizer Parameter
                param_tk = {
                    'return_tensors': "pt",
                    'padding': 'max_length',
                    'max_length': seq_len,
                    'add_special_tokens': True,
                    'truncation': True
                }
                text_batch = tokenizer_bert(text_batch, **param_tk).to(device)
                labels = torch.tensor(labels, dtype=torch.int64).to(device)

                # Feedforward prediction
                loss, logits = model(**text_batch, labels=labels)

            y_pred_tmp.extend(np.argmax(F.softmax(logits, dim=1).cpu().detach().numpy(), axis=1))
            y_truth_tmp.extend(labels.cpu().numpy())

            # Back prop
            loss.backward()

            # Training Loss
            loss_tmp += loss.item()
            loss_cnt += 1

            # Clip the gradient to prevent the exploading gradient problem in RNN/LSTM
            nn.utils.clip_grad_norm_(model.parameters(), clip)

            # Update Weights and Learning Rate
            optimizer.step()
            scheduler.step()


            #################### Evaluation ####################
            if (steps % eval_every == 0) or ((steps % eval_every != 0) and (steps == len(train_loader))):
                # Evaluate Training
                acc, f1 = metric(y_truth_tmp, y_pred_tmp)
                acc_train.append(acc)
                f1_train.append(f1)
                loss_train.append(loss_tmp/loss_cnt)
                loss_tmp, loss_cnt = 0, 0

                # y_pred_tmp = np.zeros((len(y_valid), 5))
                y_truth_tmp, y_pred_tmp = [], []

                # Move to Evaluation Mode
                model.eval()

                with torch.no_grad():
                    for i, batch in enumerate(valid_loader):
                        text_batch, meta_batch, labels = batch
                        # Skip the last batch of which size is not equal to batch_size
                        if len(labels) != batch_size:
                            break

                        if model_type == "LSTM":
                            # Tokenize the input and move to device
                            text_batch = tokenizer_lstm(text_batch, vocab, seq_len, padding='left').transpose(1,0).to(device)
                            labels = torch.tensor(labels, dtype=torch.int64).to(device)

                            # Creating new variables for the hidden state to avoid backprop entire training history
                            hidden = tuple([each.data for each in hidden])
                            for each in hidden:
                                each.to(device)

                            # Get output and hidden state from the model, calculate the loss
                            logits, hidden = model(text_batch, meta_batch, hidden)
                            loss = criterion(logits, labels)
                
                        elif model_type == 'BERT':
                            # Tokenize the input and move to device
                            text_batch = tokenizer_bert(text_batch, **param_tk).to(device)
                            labels = torch.tensor(labels, dtype=torch.int64).to(device)
                            # Feedforward prediction
                            loss, logits = model(**text_batch, **meta_batch, labels=labels)
                    
                        loss_tmp += loss.item()
                        loss_cnt += 1

                        y_pred_tmp.extend(np.argmax(F.softmax(logits, dim=1).cpu().detach().numpy(), axis=1))
                        y_truth_tmp.extend(labels.cpu().numpy())
                        # logger.debug('validation batch: {}, val_loss: {}'.format(i, loss.item() / len(valid_loader)))

                acc, f1 = metric(y_truth_tmp, y_pred_tmp)
                logging.debug("Epoch: {}/{}, Step: {}, Loss: {:.4f}, Acc: {:.4f}, F1: {:.4f}".format(epoch+1, epochs, steps, loss_tmp, acc, f1))
                acc_valid.append(acc)
                f1_valid.append(f1)
                loss_valid.append(loss_tmp/loss_cnt)
                loss_tmp, loss_cnt = 0, 0

                # Back to train mode
                model.train()

        #################### End of each epoch ####################

        # Show the last evaluation metrics
        logging.info('Epoch: %d, Loss: %.4f, Acc: %.4f, F1: %.4f, LR: %.2e' % (epoch+1, loss_valid[-1], acc_valid[-1], f1_valid[-1], scheduler.get_last_lr()[0]))

        # Plot Confusion Matrix
        y_truth_class = [class_names[int(idx)] for idx in y_truth_tmp]
        y_predicted_class = [class_names[int(idx)] for idx in y_pred_tmp]
        
        titles_options = [("Actual Count", None), ("Normalised", 'true')]
        for title, normalize in titles_options:
            disp = skplt.metrics.plot_confusion_matrix(y_truth_class, y_predicted_class, normalize=normalize, title=title, x_tick_rotation=75)
        plt.show()

        # plot training performance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))
        ax1.set_title("Losses")
        ax1.set_xlabel("Validation Cycle")
        ax1.set_ylabel("Loss")
        ax1.plot(loss_train, 'b-o', label='Train Loss')
        ax1.plot(loss_valid, 'r-o', label='Valid Loss')
        ax1.legend(loc="upper right")
        
        ax2.set_title("Evaluation")
        ax2.set_xlabel("Validation Cycle")
        ax2.set_ylabel("Score")
        ax2.set_ylim(0,1)
        ax2.plot(acc_train, 'y-o', label='Accuracy (train)')
        ax2.plot(f1_train, 'y--', label='F1 Score (train)')
        ax2.plot(acc_valid, 'g-o', label='Accuracy (valid)')
        ax2.plot(f1_valid, 'g--', label='F1 Score (valid)')
        ax2.legend(loc="upper left")

        plt.show()

        # If improving, save the number. If not, count up for early stopping
        if best_f1 < f1_valid[-1]:
            early_stop = 0
            best_f1 = f1_valid[-1]
        else:
            early_stop += 1

        # Early stop if it reaches patience number
        if early_stop >= patience:
            break

        # Prepare for the next epoch
        if device == 'cuda:0':
            torch.cuda.empty_cache()
        model.train()

    return acc, f1, model


# In[14]:


df_labels = pd.DataFrame({'label':labels})


# In[15]:


#Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(df['story'],df_labels['label'],test_size=0.2)
preprocessed = [preprocess(story) for story in df["story"]]
oversample = RandomOverSampler(sampling_strategy=0.4)
undersample = RandomUnderSampler(sampling_strategy=0.7)
train_tuples = list(zip(preprocessed, story_lengths, states, campaign_ages))
train_df = pd.DataFrame(train_tuples, columns=['story','story_length','state', 'campaign_age'])
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(train_df,df_labels['label'],test_size=0.2)
Train_X_over, Train_Y_over = oversample.fit_resample(Train_X, Train_Y)
Train_X_over, Train_Y_over = undersample.fit_resample(Train_X_over, Train_Y_over)


# In[16]:


# Define the training parameters
num_samples = [1000, 5000, 10000, 100000, 500000]
epochs=5
patience=3
batch_size=64
seq_len = 30
lr=3e-4
clip=5
log_level=logging.DEBUG

Train_X_over_text = Train_X_over['story']
Train_X_over_meta = Train_X_over[["story_length", "state", "campaign_age"]]
# Run LSTM
result_lstm, model_trained_lstm = train_cycles(Train_X_over_text, Train_X_over_meta, Train_Y_over, vocab, num_samples, 'LSTM', epochs, patience, batch_size, seq_len, lr, clip, log_level)

# Save the model and show the result
torch.save(model_trained_lstm.state_dict(), output_dir + 'campaign_lstm.dict')
result_lstm


# In[ ]:





# In[ ]:




