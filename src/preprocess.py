import re, pandas as pd
def clean_text(t):
    t=str(t).lower()
    t=re.sub(r'http\S+|www\S+','',t)
    t=re.sub(r'[^a-z ]',' ',t)
    return ' '.join(t.split())
if __name__=='__main__':
    df=pd.read_csv('data/raw/social_comments.csv')
    df['clean_text']=df['text'].apply(clean_text)
    df.to_csv('data/processed/clean_comments.csv',index=False)
    print('Saved cleaned data')