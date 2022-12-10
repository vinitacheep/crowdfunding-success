# Prediction of Surgical Crowdfunding Campaign Success using NLP

In this code, we will apply deep learning to extract word embeddings from campaign text, and then
develop a binary classification LSTM network using these word embeddings to
predict whether a campaign is able to raise its requested funds.

## Requirements
This code requires Torch. If you're on Ubuntu, installing Torch in your home directory may look something like: 

```
$ pip install torch
```

In addition, check ```import.py``` to view all necessary packages to import.  
 
 
 
 
## Implementation
### Data Preprocessing
We used data extracted from the largest crowdfunding platform, GoFundMe, from its inception
in May 2010 through December 2020 using a webscraper. This data was loaded as the dataset and processed 
using a tokenizer method. The pertinent files are found in the ```dataset``` directory. 
Specifically,  

```Clean_master_dataset.py``` loads the dataset and preprocesses.  

```Additional_dataset_filtering__silver_et_al_.py``` provides additional dataset filtration. 


### Model Definition 
For our binary text classification, we opted to utilize an LSTM Model. We tried a variety of models, found in the ```models``` directory. 
Specifically, 

```logistic_regression_and_T_SNE_plot.py``` contains the baseline logistic regression model used. Data was visualized in a t-SNE plot. 

```SVM_Naive_Bayes.py``` contains both an SVM model and a Naive Bayes model as classifiers. 

```LSTM.py``` contains the LSTM model run on purely the story text of campaigns. 

```LSTM_with_numerical_features.py``` contains the LSTM model combining story text and various numerical features as inputs.

### Loss Function
The negative log likelihood loss function script can be viewed in ```loss.py```

### Experimentation / Evaluation 
Our various model evaluations can be viewed in ```evaluate.py```

## Running scripts
To view the scripts and run the end-to-end model, run the following: 
```
$ conda env create -f environment.yml
$ conda activate cs230proj 
```
You can now load the notebook from GitHub and view. 
Runs on Linux only. May require xvfb package on headless systems.

