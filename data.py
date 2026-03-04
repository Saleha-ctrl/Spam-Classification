import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv("spam.csv",encoding="latin-1")

#removing unnecessary columns
df.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],inplace=True)

#renaming 
df.rename(columns={"v1":"label","v2":"message"},inplace=True)

#converting into 0,1
df["label"]=df["label"].map({"ham":0,"spam":1})

#checking data is balanced or not
df["label"].value_counts()

#text preprocessing

"""nltk.download("stopwords")"""
ps = PorterStemmer()

def preprocessing(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]"," ", text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words("english")]
    return " ".join(words)

df["processed_message"] = df["message"].apply(preprocessing)

#removing unprocessed message
df_copy = df.copy()
df_copy.drop(columns=["message"],inplace=True)

cv = CountVectorizer(max_features=3000)

X = cv.fit_transform(df_copy["processed_message"]).toarray()
y = df_copy["label"]


#train test split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#train model 1
model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

model_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, model_pred))
print(confusion_matrix(y_test, model_pred))
print(classification_report(y_test,model_pred))

#train model 2

model2 = MultinomialNB()
model2.fit(X_train,y_train)
model2_pred = model2.predict(X_test)

print("Accuracy:", accuracy_score(y_test, model2_pred))
print(confusion_matrix(y_test, model2_pred))
print(classification_report(y_test,model2_pred))

"""Naive Bayes has better F1-score for spam, so selected as final model."""

import pickle

# Save model
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model2, f)

# Save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(cv, f)
