import streamlit as st, joblib, pandas as pd, plotly.express as px
st.set_page_config(page_title='Sentiment Dashboard',layout='wide')
model=joblib.load('models/sentiment_model.pkl')
vec=joblib.load('models/vectorizer.pkl')

st.title('📊 Social Media Sentiment Analysis Dashboard')
tab1,tab2=st.tabs(['Single Prediction','CSV Analyzer'])

with tab1:
    txt=st.text_area('Enter comment')
    if st.button('Analyze'):
        if txt.strip():
            pred=model.predict(vec.transform([txt]))[0]
            st.success(f'Predicted Sentiment: {pred}')

with tab2:
    up=st.file_uploader('Upload CSV with text column',type='csv')
    if up:
        df=pd.read_csv(up)
        df['prediction']=model.predict(vec.transform(df['text'].astype(str)))
        st.dataframe(df.head(20),use_container_width=True)
        counts=df['prediction'].value_counts().reset_index()
        counts.columns=['sentiment','count']
        c1,c2=st.columns(2)
        with c1:
            st.plotly_chart(px.pie(counts,names='sentiment',values='count',title='Distribution'),use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(counts,x='sentiment',y='count',title='Counts'),use_container_width=True)