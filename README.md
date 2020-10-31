# Sentiment Analysis with Yelp Reviews

## Summary

I did a sentiment analysis with the review texts from Yelp. I tried two different apporaches - one is Logistic Regression with `sklearn`, and the other is Artificial Nueral Network with `TensorFlow`. The Logistic Regression model performs better and more efficiently than the ANN model in this case, with an accuracy score of 0.87 from the former model, and 0.83 from the later. It took a lot more time to fine tune the hyperparameters of the ANN model to achieve this performance. This is an instance where cooler(?) methods don't necessarily perform better, and we need to look for the most efficient means to reach our goals.

## Outline

  * [Data from Yelp](#data-from-yelp)
    + [Loading the data](#loading-the-data)
    + [Dealing with Imbalanced Reviews](#dealing-with-imbalanced-reviews)
  * [Sentiment Analysis: Identifying Five-star Reviews](#sentiment-analysis-identifying-five-star-reviews)
    + [Logsitic Regression with `sklearn`](#logistic-regression-with-sklearn)
    + [Artificial Neural Network with `TensorFlow`](#artificial-neural-network-with-tensorflow)
  * [Conclusion: Logsitic Regression Works Better!](#conclusion-logistic-regression-works-better)

## Data from [Yelp](https://www.yelp.com/dataset) 

Yelp provides data from their users free of charge, including information about reviews, business, pictures, and in different metropolitan areas. 

<img src="https://github.com/ychu19/yelp-review-sentiment-analysis/blob/main/yelp_page.jpg" width='600px'>

In this project, I did a sentiment analysis with the review texts from Yelp. I tried two different apporaches - one is Logistic Regression with `sklearn`, and the other is Artificial Nueral Network with `TensorFlow`. 

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

dt = pd.DataFrame(d)
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

rows = [Y_1, Y_2, Y_3, Y_4, Y_5]
dt = pd.concat(rows).sample(frac=1) # remember to shuffle the rows
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
`0.8776214786792522`

```python
cross_val_logit(dt.text, dt.five_stars, "auto")
```
`0.8759319792852683`

```python
cross_val_logit(dt.text, dt.five_stars, "auto")
```
`0.8736548279281595`

```python
cross_val_logit(dt.text, dt.five_stars, "auto")
```
`0.8784295001285488`

```python
cross_val_logit(dt.text, dt.five_stars, "auto")
```
`0.875895251037573`

The accuracy scores from `LogisticRegression` ranges from `0.873` to `0.878`.

### Artificial Neural Network with `TensorFlow`

#### Pre-process the data
```python
## remove stopwords

stops=" | ".join(set(stopwords.words("english")))
stops = '(?:' + stops + ')'

def custom_standardization(input_data):
  """
  1. make the texts lowercase
  2. remove periods
  3. remove stopwords
  4. remove double white space
  5. remove 's
  """
  lowercase = tf.strings.lower(input_data)
  no_periods = tf.strings.regex_replace(lowercase, '\.', '') # remove periods
  no_stop_words = tf.strings.regex_replace(no_periods, stops, ' ')
  cleaned_double_spaces = tf.strings.regex_replace(no_stop_words, '  ', ' ')
  cleaned_data = tf.strings.regex_replace(cleaned_double_spaces, '[%s]' % re.escape(string.punctuation),'')
  return cleaned_data
 
dt.text = custom_standardization(dt.text)
```

#### Split training, validation, and test sets

```python
def train_val_test(X, Y, random_state):
  """
  Split dataset into training, validation, and test sets:
  Step 1: Split training and test sets
  Step 2: Sample rows from the training set for the validation set 
  Returns train_X, train_Y, val_X, val_Y, test_X, test_Y
  """
  X = X.to_numpy()
  train_X, test_X, train_Y, test_Y = train_test_split(
      X, Y, test_size = 0.33, random_state = random_state
  )
  train_X, val_X, train_Y, val_Y = train_test_split(
      train_X, train_Y, test_size = 0.2, random_state = random_state
  )

  return train_X, train_Y, val_X, val_Y, test_X, test_Y

train_X, train_Y, val_X, val_Y, test_X, test_Y = train_val_test(dt.text, dt.five_stars, 0)
```

#### Use Pre-trained text embedding model from [TensorFlow Hub](https://tfhub.dev/)

```python
embedding_pretrained = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer=hub.KerasLayer(embedding_pretrained, input_shape = [],
                         dtype=tf.string, trainable=True)
```
The pre-trained text embedding model transforms a text vector into this:
```python
hub_layer([train_X[0]])
```
`<tf.Tensor: shape=(1, 50), dtype=float32, numpy=
array([[ 0.6559994 , -0.36600357, -0.44733047,  0.14775002, -0.08071946,
        -0.12030374,  0.5138606 ,  0.27464962, -0.87437606,  0.66484827,
         0.40583038, -0.14077562,  0.45879784,  0.36525556, -0.33027282,
        -0.37734213, -0.30459747,  0.3568462 , -0.08457891, -0.5422376 ,
        -0.37982544, -0.4337511 , -0.28198293,  0.19678144, -0.5172179 ,
         0.30254832, -0.9586079 ,  0.08863939,  0.12784152,  0.07422924,
        -0.05461134,  0.50721776,  0.676839  , -0.51586926, -0.25607902,
         0.05077176, -0.37963608,  0.42856446,  0.4734673 , -0.77815765,
         0.24654998,  0.6092979 , -0.10154548,  0.08016953, -0.1183377 ,
         0.56701434, -0.508983  , -0.3649302 ,  0.9356594 ,  0.6263341 ]],
      dtype=float32)>`

#### Bulid the Model

```python
def add_layers(model, iteration, neurons, activation, regularizer, initializer, dropout):
  """create multiple hidden layers with dropouts"""
  for i in range(iteration+1):
    model.add(tf.keras.layers.Dense(neurons, activation=activation, kernel_regularizer=regularizer, kernel_initializer=initializer))
    model.add(tf.keras.layers.Dropout(dropout))
    
dropout = 0.01

model = tf.keras.Sequential()
model.add(hub_layer)
add_layers(model, iteration=3, neurons=50, activation='elu', regularizer=None, initializer='he_normal', dropout=dropout)
model.add(layers.Dense(1, activation='sigmoid'))
```
#### Compile the Model

```python
loss_ = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics_ = tf.keras.metrics.BinaryAccuracy()
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(loss = loss_,
              optimizer=opt, # change learning rate here!
              metrics=['accuracy'])

model.save_weights('five_stars')
```

#### Train the Model

```python
model.load_weights('five_stars')

earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=3) # stops if the val_accuracy is not improving for three rounds
epochs = 15
history = model.fit(
    train_X, train_Y,
    validation_data=(val_X,val_Y),
    batch_size=200,
    epochs = epochs,
    callbacks=[earlystop])
```

#### Evaluate the Model with Test Set

```python
results = model.evaluate(test_X, test_Y, verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
```
851/851 - 1s - loss: 0.6824 - accuracy: 0.8387
loss: 0.682
accuracy: 0.839

## Conclusion: Logistic Regression Works Better (in this case)

Logistic Regression performs better than the ANN model - not only did the Logit model acheive a higher accuracy score (0.87) than the ANN model (0.84), it also took a lot more time to fine tune the hyperparameters in the ANN model to acheive such performance. I guess this is an instance where cooler(?) methods don't necessarily perform better, and we need to look for the most efficient means to reach our goals.
