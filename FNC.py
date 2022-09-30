#!/usr/bin/env python
# coding: utf-8

# In[1]:


from os import path
import pandas as pand
import numpy as nmp
import string, time, score, random, re, sys, nltk, sklearn
from nltk.corpus import stopwords
from scipy.sparse import hstack, coo_matrix, csr_matrix
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC, LinearSVC  
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as ploty
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.decomposition import PCA, TruncatedSVD, SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from keras.initializers import Constant
from keras.utils import to_categorical

# scorer & LABELS imported from FNC-1 baseline implementation
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]


print("Requirements\n")
print ("Pandas: {}".format(pand.__version__))
print ("Python: {}".format(sys.version))
print ("NLTK: {}".format(nltk.__version__))
print ("Scikit-Learn: {}".format(sklearn.__version__))
print ("Tensorflow: {}".format(tf.__version__))
print ("Numpy: {}".format(nmp.__version__))


# In[2]:


# loading the data sets
df_tokens_train = pand.read_csv("FNC_Data/FNC_Bin_Train.csv")
df_tokens_test = pand.read_csv("FNC_Data/FNC_Bin_Test.csv")
df_tokens_train['Body ID'] = df_tokens_train['Body ID'].astype(str).astype(int) 
df_tokens_train['Stance'] = df_tokens_train['Stance'].astype(str).astype(int) 
df_tokens_test['Body ID'] = df_tokens_test['Body ID'].astype(str).astype(int) 
df_tokens_test['Stance'] = df_tokens_test['Stance'].astype(str).astype(int) 


# In[3]:


df_tokens_train.head()


# In[4]:


labels_test=df_tokens_test['Stance']
labels_train= df_tokens_train['Stance']
df_labels_test= nmp.array(labels_test)
df_labels_train = nmp.array(labels_train)


# In[5]:


# hyper-parameters
vocab_length=10000
max_headline = 30
max_body = 2000


# In[6]:


# conversion of both headlines and bodies text into tokens

tokenizer = Tokenizer(num_words=vocab_length, oov_token='<OOV>')
tokenizer.fit_on_texts(df_tokens_train['Headline'].append(df_tokens_train['articleBody']))
print(tokenizer.num_words)
print(len(tokenizer.word_index))


# In[7]:


# genertation of feature matrices

train_headline_features_unpad = tokenizer.texts_to_sequences(df_tokens_train['Headline'])
train_headline_features_padded = pad_sequences(train_headline_features_unpad, maxlen=max_headline, padding='post', truncating='post')
train_body_features_unpad = tokenizer.texts_to_sequences(df_tokens_train['articleBody'])
train_body_features_padded = pad_sequences(train_body_features_unpad, maxlen=max_body, padding='post', truncating='post')
train_features = nmp.hstack([train_headline_features_padded, train_body_features_padded])

test_headline_features_unpad = tokenizer.texts_to_sequences(df_tokens_test['Headline'])
test_headline_features_padded = pad_sequences(test_headline_features_unpad, maxlen=max_headline, padding='post', truncating='post')
test_body_features_unpad = tokenizer.texts_to_sequences(df_tokens_test['articleBody'])
test_body_features_padded = pad_sequences(test_body_features_unpad, maxlen=max_body, padding='post', truncating='post')
test_features = nmp.hstack([test_headline_features_padded, test_body_features_padded])


# In[8]:


# function that takes the test train splits as inputs along with feature selection & dimension
# reduction parameters and evaluates the fit times, confusion matrices & accuracy scores for
# 4 chosen classifier models
def evaluate_classifiers (X_train, y_train, X_test, y_test, dim_red, feature_selection):
    fs='none'
    dm='none'
    
    if (feature_selection=='chi2' or feature_selection=='mi'):
        if (feature_selection=='chi2'):
            fs='Chi2 with 1000-best features'
            Kbest = SelectKBest(chi2, k=1000)
            Kbest.fit(X_train, y_train)
            X_train = Kbest.transform (X_train)
            X_test = Kbest.transform (X_test)
        if (feature_selection=='mi'):
            fs='Mutual info with 400-best features'
            Kbest = SelectKBest(mutual_info_classif, k=20)      
            Kbest.fit(X_train, y_train)
            X_train = Kbest.transform (X_train)
            X_test = Kbest.transform (X_test)
        if (dim_red=='svd'):
            dm = 'SVD redn to 300 features'
            #scaler = StandardScaler(with_mean=False).fit(X_train)
            #X_train = scaler.transform(X_train)
            #X_test = scaler.transform(X_test)
            svd = TruncatedSVD(n_components=300).fit(X_train)
            X_train = svd.transform(X_train)
            X_test = svd.transform(X_test)
        if (dim_red=='pca'):
            dm = 'PCA redn to 300 features'
            #scaler = StandardScaler(with_mean=False).fit(X_train)
            #X_train = scaler.transform(X_train)
            #X_test = scaler.transform(X_test)
            pca = PCA(n_components=300).fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)    
        
    begin = time.time()
    my_DT = DecisionTreeClassifier(criterion='entropy', random_state=0).fit(X_train, y_train)
    end = time.time()
    DT_fit_time = end-begin
    
    begin = time.time()
    my_RFC = RandomForestClassifier(n_estimators=10, random_state=50).fit(X_train, y_train)
    end = time.time()
    RFC_fit_time = end-begin
    
    begin = time.time()
    my_KN = KNeighborsClassifier().fit(X_train, y_train)
    end = time.time()
    KN_fit_time = end-begin
    
    begin = time.time()
    #my_LR = LogisticRegression(C = 1.0, class_weight='balanced', solver="lbfgs", max_iter=1000).fit(X_train, y_train)
    my_SGDC = SGDClassifier(max_iter=50, tol=None).fit(X_train, y_train)
    end = time.time()
    SGDC_fit_time = end-begin
    
    print(f"\n\nClassification Report & Confusion matrix \n(classifier= Decision tree, feature selection = {fs}, dimension reduction = {dm})\n")
    test_predictions_DT = my_DT.predict(X_test)
    print (classification_report(y_test, test_predictions_DT))
    predicted = [LABELS_RELATED[int(a)] for a in test_predictions_DT]
    actual = [LABELS_RELATED[int(a)] for a in y_test]
    print(confusion_matrix(actual, predicted, labels=['related', 'unrelated']))
    ploty.figure()
    disp = ConfusionMatrixDisplay.from_predictions(actual, predicted, cmap='YlOrBr')
    ploty.show()
    #score_DT = score.report_score(actual, predicted)
    accuracy_DT = accuracy_score(actual,predicted)
    
    print(f"\n\nClassification Report & Confusion matrix \n(classifier= Random Forest, feature selection = {fs}, dimension reduction = {dm})\n")
    test_predictions_RFC = my_RFC.predict(X_test)
    print (classification_report(y_test, test_predictions_RFC))
    predicted = [LABELS_RELATED[int(a)] for a in test_predictions_RFC]
    actual = [LABELS_RELATED[int(a)] for a in y_test]
    print(confusion_matrix(actual, predicted, labels=['related', 'unrelated']))
    ploty.figure()
    disp = ConfusionMatrixDisplay.from_predictions(actual, predicted, cmap='YlOrBr')
    ploty.show()
    accuracy_RFC = accuracy_score(actual,predicted)
    
    print(f"\n\nClassification Report & Confusion matrix \n(classifier= KNeighbor, feature selection = {fs}, dimension reduction = {dm})\n")
    test_predictions_KN = my_KN.predict(X_test)
    print (classification_report(y_test, test_predictions_KN))
    predicted = [LABELS_RELATED[int(a)] for a in test_predictions_KN]
    actual = [LABELS_RELATED[int(a)] for a in y_test]
    print(confusion_matrix(actual, predicted, labels=['related', 'unrelated']))
    ploty.figure()
    disp = ConfusionMatrixDisplay.from_predictions(actual, predicted, cmap='YlOrBr')
    ploty.show()
    accuracy_KN = accuracy_score(actual,predicted)
    
    print(f"\n\nClassification Report & Confusion matrix \n(classifier= SGDC, feature selection = {fs}, dimension reduction = {dm})\n")
    test_predictions_SGDC = my_SGDC.predict(X_test)
    print (classification_report(y_test, test_predictions_SGDC))
    predicted = [LABELS_RELATED[int(a)] for a in test_predictions_SGDC]
    actual = [LABELS_RELATED[int(a)] for a in y_test]
    print(confusion_matrix(actual, predicted, labels=['related', 'unrelated']))
    ploty.figure()
    disp = ConfusionMatrixDisplay.from_predictions(actual, predicted, cmap='YlOrBr')
    ploty.show()
    accuracy_SGDC = accuracy_score(actual,predicted)
    
    fig, ax = ploty.subplots(figsize=(10,5))
    ax2 = ax.twinx()
    ax.set_ylabel('Fit times', color='teal')
    ax2.set_ylabel('Accuracy Scores', color='coral')
    x=nmp.arange(4)
    X=['Decision Tree', 'Random Forest', 'KNeighbors', 'SGDC']
    ax.bar(x-0.1, [DT_fit_time, RFC_fit_time, KN_fit_time, SGDC_fit_time],0.2, color=['teal'], label='Fit times')
    ax2.bar(x+0.1, [accuracy_DT, accuracy_RFC, accuracy_KN, accuracy_SGDC], 0.2,  color=['coral'], label='Accuracy scores')
    ax2.set_ylim([0, 1])
    title="feature selection= "+fs+", dim reduction= "+dm
    ploty.title(title)
    ploty.xticks(x, X)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ploty.grid()
    ploty.show()
    
    return [accuracy_DT, accuracy_RFC, accuracy_KN, accuracy_SGDC]


    


# ### Classification over raw features

# In[9]:


a = evaluate_classifiers (train_features, df_labels_train, test_features, df_labels_test,'NA','NA')


# *****
# ### Classification with feature selcetion (method: Chi-2, 1000-best)

# In[10]:


b = evaluate_classifiers (train_features, df_labels_train, test_features, df_labels_test,'NA','chi2')


# ### Classification with feature selection (Chi2, 1000-best) and dimension reduction (PCA, top 300)

# In[11]:


c=evaluate_classifiers (train_features, df_labels_train, test_features, df_labels_test,'pca','chi2')


# ### Classification with feature selection (Chi2, 1000-best) and dimension reduction (SVD, 300-top)

# In[12]:


d=evaluate_classifiers (train_features, df_labels_train, test_features, df_labels_test,'svd','chi2')


# ### Comparision

# In[13]:


fig, ax = ploty.subplots(figsize=(10,5))
ax.set_ylabel('Accuracy score', color='#ff7f0e')
x=nmp.arange(4)
X=['Decision Tree', 'Random Forest', 'KNeighbors', 'SGDC']
ax.bar(x-0.3, a,0.2, color=['#17becf'], label='raw features')
ax.bar(x-0.1, b,0.2,  color=['#ff7f0e'], label='only chi2')
ax.bar(x+0.1, c,0.2, color=['#9467bd'], label='chi2+PCA')
ax.bar(x+0.3, d,0.2,  color=['#bcbd22'], label='chi2+SVD')
title="accuracy vs models vs features' choices"
ploty.title(title)
ploty.xticks(x, X)
ax.set_ylim([0, 1])
ax.legend(loc='upper right')
ploty.show()


# *****
# *****
# 
