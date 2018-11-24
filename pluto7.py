
# coding: utf-8

# In[122]:


import pandas as pd
import tensorflow as tf
import itertools
from tensorflow.python.data import Dataset
from sklearn.model_selection import train_test_split
tf.logging.set_verbosity(tf.logging.ERROR)
path = "/home/rkamath/Desktop/datascience/salesprice/"

data = pd.read_csv(path+'train.csv')
data = data.drop('Id',axis=1)


# In[123]:


def preprocess_features(data):
    selected_features = data[["OverallQual","GrLivArea","TotalBsmtSF","YearBuilt"]]
    return selected_features

FEATURES = ["OverallQual", "GrLivArea", "TotalBsmtSF", "YearBuilt"]
LABEL = "SalePrice"
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

data_pcrocessed = preprocess_features(data)
train_target = data.SalePrice
train_x, test_x, train_y, test_y = train_test_split(data_pcrocessed, train_target, test_size=0.2)

train_y = pd.DataFrame(train_y, columns = [LABEL])
training_set = pd.DataFrame(train_x, columns = FEATURES).merge(train_y, left_index = True, right_index = True)


# In[124]:


training_set.reset_index(drop = True, inplace =True)


# In[125]:


# Same thing but for the test set
test_y = pd.DataFrame(test_y, columns = [LABEL])
testing_set = pd.DataFrame(test_x, columns = FEATURES).merge(test_y, left_index = True, right_index = True)
testing_set.head()


# In[126]:


def input_fn(data_set, pred = False):
    if pred == False:
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
        labels = tf.constant(data_set[LABEL].values)
        return feature_cols, labels
    if pred == True:
        feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
        return feature_cols


# In[127]:


tf.logging.set_verbosity(tf.logging.ERROR)
regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 
                                          activation_fn = tf.nn.relu, hidden_units=[200, 100, 50, 25, 12])


# In[128]:


input_fn(training_set)


# In[129]:


regressor.fit(input_fn=lambda: input_fn(training_set), steps=2000)


# In[130]:


ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)


# In[131]:


loss_score1 = ev["loss"]
print("Final Loss on the testing set: {0:f}".format(loss_score1))

