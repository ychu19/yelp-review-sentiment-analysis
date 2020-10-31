# Sentiment Analysis with Yelp Reviews

## Summary

## Outline

  * [Data from Yelp](#data-from-yelp)
    + [Loading the data](#loading-the-data)
    + [Dealing with Imbalanced Reviews](#dealing-with-imbalanced-reviews)
  * [Sentiment Analysis: Identifying Five-star Reviews](#sentiment-analysis-identifying-five-star-reviews)
    + [Logsitic Regression with `sklearn`](#logistic-regression-with-sklearn)
    + [Artificial Neural Network with `TensorFlow`](#artificial-neural-network-with-tensorflow)
  * [Conclusion: Logsitic Regression Works Better!](#conclusion-logistic-regression-works-better)

## Data from [Yelp](https://www.yelp.com/dataset) 

### Set-up

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import re
import string
```

### Loading the data

```python
def read_json(file, max_lines):
  """Read a few lines of the json file just to take a look at the data structure"""
  count = 1
  data = []
  with open(file, 'r') as f: # 'r' means 'read'; 'w' means 'write
    for line in f:
      if count <= max_lines:
        dict_ = json.loads(line)
        dict_['five_stars'] = 1 if dict_['stars'] == 5.0 else 0
        data.append(dict_)
        count+=1
      else:
        break
  return data

path = "/content/drive/My Drive/data/yelp_academic_dataset_review.json"
d = read_json(path, 200000)
```

### Dealing with Imbalanced Reviews

I found that there were way too many 5-star reviews than others:

```python
import seaborn as sns
import matplotlib.pyplot as plt

filter_data = dt.dropna(subset=['stars'])
plt.figure(figsize=(7,4))
sns.countplot(filter_data['stars'])
```
<img src="https://github.com/ychu19/yelp-review-sentiment-analysis/blob/main/imbalanced_reviews.jpeg" width="600px" class="center">

Resampled from each group:

```python
new_n = len(dt[dt['stars']==2.0]) # 16501
Y_1 = dt[dt['stars']==1.0].sample(n=new_n, random_state=0)
Y_2 = dt[dt['stars']==2.0].sample(n=new_n, random_state=0) 
Y_3 = dt[dt['stars']==3.0].sample(n=new_n, random_state=0)
Y_4 = dt[dt['stars']==4.0].sample(n=new_n, random_state=0)
Y_5 = dt[dt['stars']==5.0].sample(n=new_n, random_state=0)
```

So that we have the same numbers of samples from each group: 
<img src="https://github.com/ychu19/yelp-review-sentiment-analysis/blob/main/adjusted_samples.jpeg" width="600px" class="center">

## Sentiment Analysis: Identifying Five-star Reviews

### Logistic Regression with `sklearn`

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression

def cross_val_logit(X, Y, multi_class):
  """
  1. randomly split training and test sets with train_test_split() 
  2. pre-process the review texts with CountVectorizer() and fit_transform the training set
  3. transform the test vector 
  4. transform with tf-idf
  5. apply logistic regression, with the option to have multinomial logit (for binary logit use "auto")
  6. return a accuracy score with the test set
  """
  train_X, test_X, train_Y, test_Y = train_test_split(
      X, Y, test_size = 0.33)
  count_vect = CountVectorizer(stop_words='english',max_df=0.85)
  train_vect = count_vect.fit_transform(train_X)
  test_vect = count_vect.transform(test_X)

  train_tf_transformer = TfidfTransformer(use_idf=True).fit(train_vect)
  train_tf = train_tf_transformer.transform(train_vect)

  test_tf = train_tf_transformer.transform(test_vect)

  log_reg = LogisticRegression(multi_class=multi_class, solver='lbfgs', max_iter=400)
  # lbfgs: "Limited-memory Broyden–Fletcher–Goldfarb–Shanno Algorithm"
  # default max_iter = 100 --> increase to 400 to allow convergence 
  log_reg.fit(train_tf, train_Y)

  return log_reg.score(test_tf, test_Y)
```
Run the function for five times for cross-validation, since the function randomly splits the training and test set with every run:

```python
cross_val_logit(dt.text, dt.five_stars, "auto")
```
0.8776214786792522

```python
cross_val_logit(dt.text, dt.five_stars, "auto")
```
0.8759319792852683

```python
cross_val_logit(dt.text, dt.five_stars, "auto")
```
0.8736548279281595

```python
cross_val_logit(dt.text, dt.five_stars, "auto")
```
0.8784295001285488

```python
cross_val_logit(dt.text, dt.five_stars, "auto")
```
0.875895251037573

The accuracy scores from `LogisticRegression` ranges from 0.873 to 0.878.

### Artificial Neural Network with `TensorFlow`

Pre-process the data

```python
## remove stopwords

stops=" | ".join(set(stopwords.words("english")))
stops = '(?:' + stops + ')'

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  no_periods = tf.strings.regex_replace(lowercase, '\.', '') # remove periods
  no_stop_words = tf.strings.regex_replace(no_periods, stops, ' ')
  cleaned_double_spaces = tf.strings.regex_replace(no_stop_words, '  ', ' ')
  cleaned_data = tf.strings.regex_replace(cleaned_double_spaces, '[%s]' % re.escape(string.punctuation),'')
  return cleaned_data

```

## Conclusion: Logistic Regression Works Better!
