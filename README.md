# Prediction of Surgical Crowdfunding Campaign Success using NLP

In this code, we will apply deep learning to extract word embeddings from campaign text, and then
develop a binary classification LSTM network using these word embeddings to
predict whether a campaign is able to raise its requested funds.

## Requirements
This code requires Torch. If you're on Ubuntu, installing Torch in your home directory may look something like: 

```
$ pip install torch
```

In addition, run the import statements in ```[import.py]``` to import all necessary packages. 
 
## Implementation
### Data Preprocessing
We used data extracted from the largest crowdfunding platform, GoFundMe, from its inception
in May 2010 through December 2020 using a webscraper. This data was loaded as the dataset and processed 
using a tokenizer method. This script can be found in ```[preprocess.py]```. 

### LSTM Model Definition 


