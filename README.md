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
using a tokenizer method. This script can be found in ```preprocess.py```. 

### LSTM Model Definition 
For our binary text classification, we opted to utilize an LSTM Model, which is defined in ```LSTM_model.py```

### Loss Function
The negative log likelihood loss function script can be viewed in ```loss.py```

### Experimentation / Evaluation 
Our various model evaluations can be viewed in ```evaluate.py```

## Jupyter Notebook
To view the scripts in Jupyter Notebook and run the end-to-end model, run the following: 
```
$ conda env create -f environment.yml
$ conda activate cs230proj 
```
Then, run ```$ jupyter notebook``` to open the Jupyter Notebook webpage. You can now load the notebook from GitHub and view. 
Runs on Linux only. May require xvfb package on headless systems.

