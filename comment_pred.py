import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
stop_words=set(stopwords.words('english'))
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,roc_auc_score,RocCurveDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
import pickle
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import mysql.connector
import googleapiclient.discovery
import time as t
import time
import re
import streamlit as st
from streamlit_option_menu import option_menu


with open ("model_1.pkl","rb") as files:
    tf_model = pickle.load(files)

def Api_connect():
    
    api_key = "AIzaSyATGv2bIbANrgYutI_H2ymikbLrepMbkHk"

    api_service_name = "youtube"
    api_version = "v3"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version,developerKey = api_key)
    return youtube

youtube = Api_connect()

def get_channel_details(chan_id):
    request = youtube.channels().list(
    part="snippet,contentDetails,statistics",
    id=chan_id)
    response=request.execute()
    
    channel_details = dict(channel_id = response['items'][0].get('id', "data doesn't exist"),channel_name = response['items'][0]['snippet'].get('title',"Title doesn't exist"),
    channel_description = response['items'][0]['snippet'].get('description',"Description doesn't exist"),Subscribers_count = response['items'][0]['statistics'].get('subscriberCount',"subscriberCount doesn't exist"),
    Views_count = response['items'][0]['statistics'].get('viewCount',"View_counts don't exist"),Likes_count = response['items'][0]['contentDetails'].get('likes',"Likes don't exist"))
    channel_df = pd.DataFrame([channel_details],index=[1])
    return channel_df

def get_play_lists(chan_id):
    nextPageToken = None
    all_play_lists = []
    while True:
        request = youtube.playlists().list(
                part="snippet,contentDetails",
                channelId=chan_id,
                maxResults=25,
                pageToken=nextPageToken)
        response = request.execute()
        all_play_lists.extend(response['items'])
        nextPageToken = response.get('nextPageToken')
        if nextPageToken == None:
                break
            
    return all_play_lists

def dic_conv(chan_id):
    all_play_lists = get_play_lists(chan_id)
    new_dic = {}
    for d in all_play_lists:
        for k, v in d.items():
            new_dic[k] = new_dic.get(k, []) + [v]
    return new_dic

def play_list_dframe(chan_id):
    txt_1 = dic_conv(chan_id)
    new_dic = txt_1
    ch_name = []
    for i in new_dic['snippet']:
        ch_name.append(i['channelTitle'])
    chan_id = []
    for i in new_dic['snippet']:
        chan_id.append(i['channelId'])
    v_c = []
    for i in new_dic['contentDetails']:
        v_c.append(i['itemCount'])

    play_dic = {"Channel_name":ch_name,
            "Channel_id" : chan_id,"Play_list_id" : new_dic.get('id','Data not available'), 
            'Play_list_title':[i['title'] for i in new_dic['snippet']],'Video_count' : v_c,
            'Pub_date':[i['publishedAt'] for i in new_dic['snippet']]}
    play_list_df = pd.DataFrame(play_dic,index=range(1,len(ch_name)+1))
    return play_list_df

def get_video_id(chan_id):

    request = youtube.channels().list(
    part="snippet,contentDetails,statistics",
    id=chan_id
    )
    response=request.execute()
    
    Play_list = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    Play_list
    play_id=Play_list
    
    vid_request = youtube.playlistItems().list(
        part='contentDetails',
        playlistId = play_id,
        maxResults=100       
    )
    response = vid_request.execute()
    videos_id=[]
    for i in range(len(response['items'])):
        videos_id.append((response['items'][i]['contentDetails']['videoId']))

    nextPageToken = response.get('nextPageToken')
    
    while True:    
        if nextPageToken is None:
            break
        else:
            vid_request = youtube.playlistItems().list(
            part='contentDetails',
            playlistId = play_id,
            maxResults=50,
            pageToken = nextPageToken
            )
            response = vid_request.execute()

            for i in range(len(response['items'])):
                videos_id.append((response['items'][i]['contentDetails']['videoId']))
            
            nextPageToken = response.get('nextPageToken')

    #video_id_df = pd.DataFrame(videos_id,columns=['Video_ids'],index=range(1,len(videos_id)+1))   
       
    return videos_id

def get_video_info(chan_id):
        videos_id = get_video_id(chan_id)
        responses = []
        for video_id in videos_id :
                request1 = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_id)        
                response1 = request1.execute()
                responses.append(response1)

        video_data =[]        
        for response in responses:
                for video in response['items']:
                        video_stat = dict(channel_id=video['snippet']['channelId'],channel_Name=video['snippet']['channelTitle'],Title = video['snippet']['title'],Published_date=video['snippet']['publishedAt'],video_description =video['snippet']['description'],
                                view_count = video['statistics']['viewCount'],Like_count = video['statistics'].get('likeCount'),
                                favorite_count = video['statistics']['favoriteCount'],comment_count = video['statistics'].get('commentCount'),
                                Video_id = video['id'],play_list_id = video['snippet']['channelId'])
                        
                        video_data.append(video_stat)
        df = pd.DataFrame(video_data)
        return df.head(5)

def get_comments(chan_id):
    videos_id = get_video_id(chan_id)
    comment_info=[]
    try:
        for i in videos_id:
            comm_request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=i        
            )
            response = comm_request.execute()
            for comment in response['items']:
                comment_details = dict(channel_id = comment['snippet']['channelId'],video_id = comment['snippet']['videoId'],
                          comment_text = comment['snippet']['topLevelComment']['snippet']['textDisplay'],
                          comment_author = comment['snippet']['topLevelComment']['snippet']['authorDisplayName'],
                          comment_published_date = comment['snippet']['topLevelComment']['snippet']['publishedAt'],
                          comment_id = comment['snippet']['topLevelComment']['id'])      
                comment_info.append(comment_details)
    except:
         pass
    return pd.DataFrame(comment_info)

def get_comments(video_id):
    comment_info=[]
    try:
        next_page_token = None
        while True:
            comm_request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token)
            response = comm_request.execute()            
            if not response.get('items','No key found'):
                    #print(f"{i} :{'No comments found'}")
                    continue
            else:
                for comment in response['items']:
                    comment_details = dict(channel_id = comment['snippet']['channelId'],video_id = comment['snippet']['videoId'],
                            comment_text = comment['snippet']['topLevelComment']['snippet']['textDisplay'],
                            comment_author = comment['snippet']['topLevelComment']['snippet']['authorDisplayName'],
                            comment_published_date = comment['snippet']['topLevelComment']['snippet']['publishedAt'],
                            comment_id = comment['snippet']['topLevelComment']['id'])      
                    comment_info.append(comment_details)
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
    except:
         pass
    
    return  pd.DataFrame(comment_info)     
    
    

def get_video_titles(chan_id):
    video_details = get_video_info(chan_id)
    video_titles = video_details[['Title','Video_id']]
    return video_titles


def cleaned_text(texts):
    clean_texts =[]
    for text in texts:
        text = text.lower()
        tokens = word_tokenize(text)
        cleaned_token =[]
        lemmatizer = WordNetLemmatizer()
        for word in tokens:
            if (word.isalnum()) and (not word.isdigit()) and (word not in stop_words):
                word = word.strip(string.punctuation)
                word = lemmatizer.lemmatize(word)
                word = word.strip()
                cleaned_token.append(word)
            cleaned_text = " ".join(cleaned_token)
        clean_texts.append(cleaned_text if cleaned_text else "no meaningful words")
    return clean_texts

def get_sentiments(texts):
    sentiments = []
    for text in texts:
        analysis = TextBlob(text)
        if analysis.sentiment.polarity>0:
            sentiments.append("Positive")
        else:
            sentiments.append('Negative')
    return sentiments

def get_cleaned_data(video_id):
    comment_df = get_comments(video_id)
    if comment_df.empty:
        st.header('No comments found')
    
            
    else:
        corpus = comment_df['comment_text'].values
        comment_df['cleaned_comment'] = cleaned_text(corpus)
        corpus_1 = comment_df['cleaned_comment'].values
        comment_df['sentiment'] = get_sentiments(corpus_1)
        comment_df['SENTIMENTS'] = comment_df['sentiment'].apply(lambda x:1 if x == 'Positive' else 0)
        return comment_df


def get_result(video_id):
    vid_id = video_id
    comment_df = get_cleaned_data(vid_id)  
    model = TfidfVectorizer()
    if comment_df.empty:
        st.header('No comments found')
                
    else:
        corpus = comment_df['cleaned_comment'].values
        model.fit(corpus)
        data = pd.DataFrame(model.fit_transform(corpus).toarray())
        data.columns = model.get_feature_names_out()
        data['Toxicity'] = comment_df['sentiment']
        X= data.drop(['Toxicity'],axis=1)
        y = data['Toxicity']
        z = y.to_string(index=False)
        cls_lst = z.split()
        a = cls_lst.count('Negative')
        b = cls_lst.count('Positive')
        c = len(cls_lst)
        if a == c:
            return 'Negative comments only'
        elif b == c:
            return "Positive comments only"            

        else:
            mdl = LogisticRegression()
            mdl.fit(X,y)           
            test_text = [comment_df['cleaned_comment'].loc[0]]
            test_vector = model.transform(test_text)
            pred = mdl.predict(test_vector)
            return f"Predicted Sentiment: {pred[0]}"
        
def comment_count(video_id):
    vid_id = video_id
    comment_df = get_cleaned_data(vid_id)
    if isinstance(comment_df,str):
        return "No comments found"
    elif not comment_df.empty:
        return 'No comments found'
    else:
        com_count = comment_df[comment_df['video_id'] == vid_id]['comment_text'].count()
        return com_count
        
def comment_plot(video_id):
    vid_id = video_id
    comment_df = get_cleaned_data(vid_id)
    if isinstance(comment_df,str):
        st.header("No comments found")
        
    else:
        fig,ax = plt.subplots(figsize=(10,8))
        ax.bar = sns.countplot(data=comment_df,x='video_id',hue='sentiment',legend=True,palette={"Positive": "green", "Negative": "red"})
        ax.set_title('SENTIMENTS COUNT')
        try:
            ax.bar_label(container = ax.containers[0])
            ax.bar_label(container= ax.containers[1])
        except:
            pass
        plt.close(fig)
        return fig
    
def comment_pie_plot(video_id):
    vid_id = video_id
    comment_df = get_cleaned_data(vid_id)
    if isinstance(comment_df,str):
        st.header("No comments found")
        
    else:
        fig,ax = plt.subplots(figsize=(2,2))
        comment_df['sentiment'].value_counts().plot(kind='pie',autopct='%1.1f%%')
        ax.set_title('Type of comments count')
        plt.close(fig)
        return fig
    
def result_df(chan_id):
    df = get_video_titles(chan_id)
    df =df.head(7)    
    result = [[v['Video_id'], v['Title'],get_result(v['Video_id'])] for k, v in df.iterrows()]
    df_1 = pd.DataFrame(result,columns=['VIDEO_ID','VIDEO_TITLE','RESULT'])
    return df_1

st.set_page_config(layout="wide")

st.set_option('deprecation.showPyplotGlobalUse', False)

title_text = '''<h1 style='font-size: 55px;text-align: center;color:purple;background-color: lightgrey;'>Sentiment Analysis'''
st.markdown(title_text, unsafe_allow_html=True)

with st.sidebar:

    select = option_menu("MANIN MENU",['CHANNELS','VIDEO','PREDICTION'])


if select == 'CHANNELS':

    chan_id = st.text_input('Enter the channel_id:herb:')

    if st.button("Extract Data:satellite_antenna:"):

        with  st.spinner("Collecting Data...:running:"):
            t.sleep(8)

        chan_details = get_channel_details(chan_id)
        st.title("Channel information:loudspeaker:")
        st.dataframe(chan_details)

        with st.spinner("Collecting Data....:running:"):
            t.sleep(8)

        play_list_details = play_list_dframe(chan_id)
        st.title('Playlists Information:loudspeaker:')
        st.dataframe(play_list_details)

if select == 'VIDEO':

        chan_id = st.text-input('Enter the channel id:herb:')
        
        if st.button("Submit:satellite_antenna"):

            with st.spinner("Collecting Data...:running:"):
                t.sleep(8)

            video_details = get_video_info(chan_id)
            st.title("Video Information:loudspeaker:")
            st.dataframe(video_details)
        
            comments_details = get_comments(chan_id)
            st.title('Comments Information:loudspeaker')
            if isinstance(comments_details,str):
                st.write('No comments found')
            else:
                st.dataframe(comments_details)

if select == 'PREDICTION':

        chan_id = st.text_input('Enter the channel id:herb:')

        if st.button('GET:satellite_antenna:'):

            with st.spinner("Wait for it..."):
                t.sleep(8)

            video_titles = get_video_titles(chan_id)
            st.title('Video Titles Information:loudspeaker:')
            st.dataframe(video_titles)

            with st.spinner("Wait for it..."):
                t.sleep(8)

            pred_details = result_df(chan_id)
            st.title('Prediction Details:loudspeaker:')
            st.dataframe(pred_details)

        video_id = st.text_input('Enter the video_id:herb:')

        if st.button('Submit:satellite_antenna:'):

            with st.spinner("Wait for it..."):
                t.sleep(8)

            com_details = get_comments(video_id)
            st.header('Comments details:loudspeaker:')
            st.dataframe(com_details)
            comment_details = get_cleaned_data(video_id)
            st.header('Cleaned Comments details:loudspeaker:')
            st.dataframe(comment_details)

            prediction = get_result(video_id)
            st.header('Prediction details:loudspeaker:')
            st.header(prediction)

            st.write("  ")
            st.write("  ")
            st.write(" ")

            col1,col2 = st.columns(2)

            with col1:
                comment_bar_plot = comment_plot(video_id)
                if isinstance(comment_bar_plot,str):
                    st.write('No comments found')
                else:
                    st.pyplot(comment_bar_plot)
            with col2:
                comment_pie_plots = comment_pie_plot(video_id)
                if isinstance(comment_pie_plots,str):
                    st.write('No comments found')
                st.pyplot(comment_pie_plots)

