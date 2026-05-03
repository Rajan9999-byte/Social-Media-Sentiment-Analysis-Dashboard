import os, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import clean_text

raw='data/raw/social_comments.csv'
df=pd.read_csv(raw)
df['clean_text']=df['text'].apply(clean_text)
os.makedirs('data/processed',exist_ok=True)
df.to_csv('data/processed/clean_comments.csv',index=False)

X_train,X_test,y_train,y_test=train_test_split(df['clean_text'],df['label'],test_size=0.2,random_state=42,stratify=df['label'])
vec=TfidfVectorizer(max_features=3000,ngram_range=(1,2))
Xtr=vec.fit_transform(X_train); Xte=vec.transform(X_test)
model=LogisticRegression(max_iter=1000)
model.fit(Xtr,y_train)
pred=model.predict(Xte)
print('Accuracy:',round(accuracy_score(y_test,pred),4))
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
os.makedirs('models',exist_ok=True)
joblib.dump(model,'models/sentiment_model.pkl')
joblib.dump(vec,'models/vectorizer.pkl')
print('Saved model files')