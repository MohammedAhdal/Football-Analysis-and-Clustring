import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import joblib  # To load the trained model

# Function to load data (modify as needed)
def load_data():
    df = pd.read_csv('team_cluster.csv')
    return df

# Function to load the trained model
def load_model():
    model = joblib.load('kmeans2_model.pkl')  # Replace with your model's filename
    return model

# Load your data and model
df = load_data()
model = load_model()
# Streamlit app layout
st.title('Football Team Performance Clustering & Analysis')

# sidebar
st.sidebar.header("Football Performances Analysis")
st.sidebar.image('football.jpg')
st.sidebar.write("This dashboard is using 2021-2022 Football Stats datasets as part of my final project for the Data Science Bootcamp")
st.sidebar.write("")
st.sidebar.subheader("Filter your data")
cat_filter = st.sidebar.selectbox("Categorical Filtering", [None, 'Squad', 'Country', 'Cluster'])
num_filter = st.sidebar.selectbox("Numerical Filtering", [None, 'W', 'Pts/G', 'Prog_Passes', 'Shots_OT', 'Carries_3rd', 'Tackles_Won', 'Crosses', 
                                                       'Aerials_Won', 'Interceptions', 'Clearances', 'Blocks'])
#row_filter = st.sidebar.selectbox("Row Filtering", ['Squad', 'Country', 'Cluster'])
#col_filter = st.sidebar.selectbox("Columns Filtering", ['W', 'GD', 'Pts/G', 'xGD/90'])

show_data = st.sidebar.checkbox('Show Raw Data')
show_pie_chart = st.sidebar.checkbox('Show Pie Chart')
show_bar_chart = st.sidebar.checkbox('Show Bar Chart')


st.sidebar.markdown("Made with :heart_eyes: by [Mohammed Al Ahdal]('https://mohammedahdal.github.io/portfolio/')")

# body area

# row a
a1, a2, a3 = st.columns(3)

a1.metric("Number of Teams", df['Squad'].count())
a2.metric("Top Shots on Target", df['Shots_OT'].max())
a3.metric("Max Wins", df['W'].max())

if show_data:
    st.write(df)


# row b
fig = px.scatter(data_frame=df,
                x=num_filter,
                y='W',
                color=cat_filter,
                size=num_filter,
                #facet_col=col_filter,
                #facet_row=row_filter,
                #facet_row_spacing = 0.03
                )
st.plotly_chart(fig, use_container_width=True)

# Pie Chart
if show_pie_chart:
    pie_data = df['Country'].value_counts()  # Replace with your column
    pie_fig = px.pie(pie_data, values=pie_data.values, names=pie_data.index, title='Pie Chart')
    st.plotly_chart(pie_fig)

# Bar Chart
if show_bar_chart:
    bar_data = df['Own_Goals'].value_counts()  # Replace with your column
    bar_fig = px.bar(bar_data, x=bar_data.index, y=bar_data.values, title='Bar Chart')
    st.plotly_chart(bar_fig)