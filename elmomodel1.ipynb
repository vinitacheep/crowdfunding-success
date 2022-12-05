{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import spacy\n",
    "import re\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pip install \"tensorflow>=2.0.0\"\n",
    "pip install --upgrade tensorflow-hub"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "columns = [\"raised_amnt\", \"goal_amnt\", \"story\"]\n",
    "df = pd.read_csv('small_filtered_geotagged_surgery_campaigns.csv', usecols=columns, encoding='utf8')\n",
    "print(\"Contents in csv file: \", df)\n",
    "print(df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow_hub as hub\n",
    "import tensorflow.compat.v1 as tf\n",
    "tf.disable_eager_execution()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "%time elmo = hub.Module(\"https://tfhub.dev/google/elmo/2\", trainable=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def elmo_vectors( stories ):\n",
    "    embeddings = elmo(stories.tolist(), signature=\"default\", as_dict=True)[\"elmo\"]\n",
    "    with tf.Session() as sess:\n",
    "        sess.run(tf.global_variables_initializer())\n",
    "        sess.run(tf.tables_initializer())\n",
    "        # average along axis 1\n",
    "        return sess.run(tf.reduce_mean(embeddings,1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "batch_size = 4\n",
    "list_train = [df[i:i+batch_size] for i in range(0,df.shape[0], batch_size)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "elmo_train = [elmo_vectors(x[\"story\"]) for x in list_train]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "elmo_train_new = np.concatenate(elmo_train, axis=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(elmo_train_new)\n",
    "print(elmo_train_new.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pd.DataFrame(elmo_train_new).to_csv('encodings.csv') "
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
