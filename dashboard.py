# app/dashboard.py

import streamlit as st
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
from datetime import datetime, time

import matplotlib.pyplot as plt
import pytz
warnings.filterwarnings('ignore')

# Set up the Streamlit page
st.set_page_config(page_title="Twitter Dashboard", layout="wide")

# Title of the app
st.title("📊 Twitter Analytics Dashboard")

# Load the tweet data from the CSV
df = pd.read_excel(r"F:\twitterdashboard\data\Tweet.xlsx")

# Show the first few rows
st.subheader("Raw Tweet Data")
st.dataframe(df.head())
##1. Create a visual that shows the average engagement rate and total impressions for tweets
##posted between '01- 01-2020' and '30-06-2020'. Filter out tweets that received fewer than 100
##impressions and like should be 0 and this graph should work only between 3PM IST to 5 PM IST
##apart from that time we should not show this graph in dashboard itself

df['time']=pd.to_datetime(df['time'],errors='coerce',utc=True)
start_date = pd.to_datetime('2020-01-01').date()
end_date = pd.to_datetime('2020-06-30').date()

# Filter the DataFrame using the converted date objects
df7 = df[(df["impressions"] > 100) & (df["likes"] == 0) & df["time"].dt.date.between(start_date, end_date)]
# Current time in IST


avg_engagement_rate=df7['engagement rate'].mean()
sum_of_impression=df7['impressions'].sum()
data_for_plot = {'Metric': ['Average Engagement Rate', 'Total Impressions'],
                 'Value': [avg_engagement_rate, sum_of_impression]}
df_plot = pd.DataFrame(data_for_plot)

fig = px.bar(df_plot, x='Metric', y='Value', title="Average Engagement Rate and Total Impressions")
st.plotly_chart(fig)


##2. Plot a scatter chart to analyse the relationship between media engagements and media views for tweets that received more than 10 replies.
## Highlight tweets with an engagement rate above 5% and this graph should work only between 6PM IST to 11 PM IST apart from that time we
##should not show this graph in dashboard itself and the tweet date should be odd number as well as tweet word count be above 50.
df['word_count']=df['Tweet'].apply(lambda x:len(x.split()))
df8 = df[
    (df["replies"] > 10) &
    (df["engagement rate"] > 0.05) &
    (df["time"].dt.day % 2 != 0) &
    (df["word_count"] > 50)
]

fig = px.scatter(
        df8,
        x="media views",
        y="media engagements",
        color="engagement rate",
        hover_data=["Tweet", "replies"],
        title=" Media Engagements vs. Views (High Engagement, Odd Dates, Long Tweets)"
    )
fig.update_layout(template="plotly_white")
st.plotly_chart(fig)
    

df['Date']=df["time"].dt.date
df["length"]=df['Tweet'].apply(lambda x:len(x))    


##3. Create a clustered bar chart that breaks down the sum of URL clicks, user profile clicks, and hashtag clicks by tweet category
 ##(e.g., tweets with media, tweets with links, tweets with hashtags). Only include tweets that have at least one of these interaction types
 ##and this graph should work only between 3PM IST to 5 PM IST apart from that time we should not show this graph in dashboard itself and
 ##the tweet date should be even number as well as tweet word count be above 40.
def categorize_tweet(row):
    if row['media views'] > 0:
        return 'Media'
    elif row['url clicks'] > 0:
        return 'Link'
    elif row['hashtag clicks'] > 0:
        return 'Hashtag'
    else:
        return 'Other'

df['Tweet Category'] = df.apply(categorize_tweet, axis=1)

# Filter for required conditions
df1 = df[
    (df['length'] > 40) &
    (df['time'].dt.day % 2 == 0) &
    ((df['url clicks'] > 0) | (df['user profile clicks'] > 0) | (df['hashtag clicks'] > 0))
]

# Group by Tweet Category
grouped = df1.groupby('Tweet Category')[['url clicks', 'user profile clicks', 'hashtag clicks']].sum().reset_index()

# Melt for plot
melted = grouped.melt(
    id_vars='Tweet Category',
    value_vars=['url clicks', 'user profile clicks', 'hashtag clicks'],
    var_name='Click Type',
    value_name='Total Clicks'
)

# Plot using Plotly
import plotly.express as px
fig = px.bar(
    melted,
    x='Tweet Category',
    y='Total Clicks',
    color='Click Type',
    barmode='group'  # or 'stack' for stacked bar chart
)
st.plotly_chart(fig)

##4. Build a chart to identify the top 10 tweets by the sum of retweets and likes. Filter out
##tweets posted on weekends and show the user profile that posted each tweet and this
##graph should work only between 3PM IST to 5 PM IST apart from that time
##we should not show this graph in dashboard itself and the tweet impression
##should be even number and tweet date should be odd number as well as tweet word count be below 30
df2=df[(df['impressions']%2==0) & (df['time'].dt.day %2!=0) & (df['word_count']<30) & (df['time'].dt.weekday <5)]
df2["sum_of_retweet_likes"]=df2["retweets"]+df2["likes"]
df3=df2.sort_values(by="sum_of_retweet_likes",ascending=False).head(10)
df3["user_profile"]=[f"user {i+1}"for i in range(len(df3))]
fig = px.bar(df3, x="user_profile", y="sum_of_retweet_likes", title="Top 10 Tweets by Retweets and Likes")
st.plotly_chart(fig)
  
##5. Create a dual-axis chart that shows the number of media views and media engagements by the day of the week for the last quarter.
##Highlight days with significant spikes in media interactions. this graph should work only between 3PM IST to 5 PM IST and 7 AM to 11AM
##apart from that time we should not show this graph in dashboard itself and the tweet impression should be even number and tweet date
##should be odd number as well as tweet character count should be above 30 and need to remove tweet word which has letter 'H'.
df["char_count"]=df["Tweet"].apply(lambda x:len(x))
df10=df[(df["impressions"]%2==0) & (df["time"].dt.day %2!=0) & (df["char_count"]>30)]
df10["Tweet"]=df10["Tweet"].apply(lambda x : ' '.join([word for word in x.split() if "h" not in word.lower()]))
from plotly.subplots import make_subplots
import plotly.graph_objects as go
df10["day_of_week"] = df10["time"].dt.day_name()

grouped=df10.groupby("day_of_week").agg({"media views":"sum","media engagements": "sum"})



fig = make_subplots(specs=[[{"secondary_y": True}]])
# Primary Y-axis: Media Views
fig.add_trace(
     go.Bar(x=grouped.index, y=grouped["media views"], name="Media Views", marker_color='lightskyblue'),
     secondary_y=False
)


fig.add_trace(
     go.Scatter(x=grouped.index, y=grouped["media engagements"], name="Media Engagements", mode='lines+markers'),
     secondary_y=True
    
)
st.plotly_chart(fig)


df5=df[(df['time'].dt.day %2!=0) & (df['engagements']%2==0) & (df['length']>20)]
df5['tweet_without_c']=df5["Tweet"].apply(lambda x: ''.join([word for word in  x.split() if 'c' not in word.lower()]))
df5['month']=df['time'].dt.month


# --- 2. Determine if a tweet has media ---
df5['has_media'] = df5['media views'].fillna(0) + df5['media engagements'].fillna(0) > 0
df5['media_type'] = df5['has_media'].apply(lambda x: 'With Media' if x else 'Without Media')
monthly_avg = df5.groupby(['month', 'media_type'])['engagement rate'].mean().reset_index()
# --- 3. Extract month for grouping ---
df['month'] = df['time'].dt.month
ist = pytz.timezone('Asia/Kolkata')
current_time = datetime.now(ist).time()
if time(15,0)<=current_time<=time(17,0) or time(19,0)<=current_time<=time(21,0):

    fig=px.line(monthly_avg,x="month",y="engagement rate",color='media_type',markers=True,
               title='Monthly Average Engagement Rate (With vs Without Media)',labels={
            'month': 'Month',
            'engagement rate': 'Avg Engagement Rate',
            'media_type': 'Tweet Type'
        })
    st.plotly_chart(fig)

else:
    st.warning("chart should be dispalyed between 3pm to 5 pm and 7pm to  9pm")

##7. Analyse tweets to show a comparison of the engagement rate for tweets with
##app opens versus tweets without app opens. Include only tweets posted between 9 AM and 5 PM on
##weekdays andthis graph should work only between 12PM IST to 6PM IST and 7 AM to 11AM apart from
##that time we should not show this graph in dashboard itself and the tweet impression
##should be even number and tweet date should be odd number as well as tweet character count
##should be above 30 and need to remove tweet word which has letter 'D'

# Define time window
start_time = time(9, 0)
end_time = time(17, 0)

# Step 1: Filter data based on all criteria
df6 = df[
    (df['impressions'] % 2 == 0) &
    (df['time'].dt.day % 2 != 0) &
    (df['length'] > 30) &
    (df['time'].dt.weekday < 5) &
    (df['time'].dt.time >= start_time) &
    (df['time'].dt.time <= end_time)
]

# Step 2: Clean tweets by removing words with 'D' or 'd'
df6["tweet_without_d"] = df6['Tweet'].apply(
    lambda x: ' '.join([word for word in x.split() if 'd' not in word.lower()])
)

# Step 3: Calculate engagement rate
df6['engagement_rates'] = df6['engagements'] / df6['impressions']

# Step 4: Add app open status
df6['has_app_open'] = df6['app opens'] > 0

# Step 5: Group by app open and calculate mean engagement rate
summary = df6.groupby('has_app_open')['engagement_rates'].mean().reset_index()
summary['App Opens'] = summary['has_app_open'].map({True: 'With App Opens', False: 'Without App Opens'})

# Step 6: Time-based control (IST)
ist = pytz.timezone('Asia/Kolkata')
now_ist = datetime.now(ist).time()

show_graph = (time(12, 0) <= now_ist <= time(18, 0)) or (time(7, 0) <= now_ist <= time(11, 0))

# Step 7: Show or hide graph
if show_graph:
    fig = px.bar(
        summary,
        x='App Opens',
        y='engagement_rates',
        color='App Opens',  # Optional: for colored bars
        title='Engagement Rate: With vs Without App Opens'
    )
    st.plotly_chart(fig)
else:
    st.warning("Graph hidden due to time restriction.")