import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
st.title('Decoding The Beautiful Game')
st.write("")
# sidebar
st.sidebar.header("Football Performances Analysis")
st.sidebar.image('football.jpg')
st.sidebar.write("This dashboard is using 2021-2022 Football Stats datasets as part of my final project for the Data Science Bootcamp")
st.sidebar.markdown("Made with :heart_eyes: by [Mohammed Al Ahdal]('https://mohammedahdal.github.io/portfolio/')")
st.sidebar.write("- - -")

# Specific metrics for selection
specific_metrics = ['Prog_Passes', 'Shots_OT', 'Carries_3rd', 'Tackles_Won', 'Crosses', 'Aerials_Won', 'Clearances', 'Blocks', 'Offsides', 'Own_Goals', 'Yellow_Cards', 'Red_Cards']
# Replace 'ENG' with 'GBR' for plotting, but keep 'ENG' as a label
df['DisplayCountry'] = df['Country'].replace({'GBR': 'ENG'})
df['Country'] = df['Country'].replace({'ENG': 'GBR'})

# Sidebar for user inputs
st.sidebar.header("Exploratory Data Analysis")
selected_metric = st.sidebar.selectbox("Select Metric for EDA", specific_metrics, index=0, key='eda_metric')

# Combined EDA plot option
show_histogram_boxplot = st.sidebar.checkbox('Show Histogram & Boxplot', key='show_histogram_boxplot')

# Display EDA plots in a row with two columns

if show_histogram_boxplot:
    st.write("- - -")
    st.subheader('EDA Visualisation')
    col1, col2 = st.columns(2)
    with col1:
        # st.subheader(f'Histogram of {selected_metric}')
        hist_fig = px.histogram(df, x=selected_metric, nbins=20, title=f'Histogram of {selected_metric}', color_discrete_sequence=['indianred'])
        st.plotly_chart(hist_fig, use_container_width=True)

    with col2:
        # st.subheader(f'Boxplot of {selected_metric}')
        box_fig = px.box(df, y=selected_metric, title=f'Boxplot of {selected_metric}', color_discrete_sequence=['lightseagreen'])
        st.plotly_chart(box_fig, use_container_width=True)

# Correlation Heatmap
st.write("- - -")
show_correlation_heatmap = st.sidebar.checkbox('Show Correlation Heatmap', key='show_correlation_heatmap')
if show_correlation_heatmap:
    st.subheader('Correlation Heatmap')
    numeric_cols = df[specific_metrics].select_dtypes(include=['float64', 'int64'])
    corr = numeric_cols.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)


# Clustering Results Visualization

st.sidebar.write("- - -")
if st.sidebar.checkbox('Show Clustering Results'):
    
    # Selecting features to plot
    cat_filter = st.sidebar.selectbox("Categorical Filtering", [None, 'Squad', 'Country', 'Cluster'])
    num_filter = st.sidebar.selectbox("Numerical Filtering", [None, 'W', 'Pts/G', 'Prog_Passes', 'Shots_OT', 'Carries_3rd', 'Tackles_Won', 'Crosses', 
                                                       'Aerials_Won', 'Interceptions', 'Clearances', 'Blocks'])
    
    # Create and display the scatter plot
    st.write("- - -")
    st.subheader('Clustering Results Visualization')
    fig = px.scatter(data_frame=df,
                x=num_filter,
                y='W',
                color=cat_filter,
                size=num_filter
                )
    st.plotly_chart(fig, use_container_width=True)


# Sidebar for Country Analysis user inputs
st.sidebar.write("- - -")
st.sidebar.header("Country Analysis")
selected_metric = st.sidebar.selectbox("Select Metric for Analysis", specific_metrics, index=0, key='country_analysis_metric')

# Generate and display charts
if st.sidebar.button('Show Country Charts'):
    # Aggregate data for countries
    country_agg_data = df.groupby('DisplayCountry')[selected_metric].sum().reset_index()

    # Create Pie Chart
    pie_fig = px.pie(country_agg_data, values=selected_metric, names='DisplayCountry', title=f'Pie Chart of {selected_metric} by Country')
    
    # Create Horizontal Bar Chart
    bar_fig = px.bar(country_agg_data, y='DisplayCountry', x=selected_metric, title=f'Horizontal Bar Chart of {selected_metric} by Country', orientation='h', color='DisplayCountry')
    bar_fig.update_layout(showlegend=False)

    # Create Choropleth Map using the official country codes
    map_fig = px.choropleth(df, locations='Country',
                            locationmode='ISO-3',
                            color=selected_metric,
                            title=f'Map of {selected_metric} by Country',
                            color_continuous_scale=px.colors.sequential.Plasma)
    # Set the map's geographical boundaries to focus on Western Europe
    map_fig.update_geos(
        visible=False, resolution=50,
        lonaxis_range=[-10,20],  # Adjust the longitude range as needed
        lataxis_range=[35, 60]    # Adjust the latitude range as needed
    )
    st.write("- - -")
    st.header("Country Analysis")
    # Display Pie Chart and Choropleth Map side by side
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(pie_fig, use_container_width=True)
    with col2:
        st.plotly_chart(map_fig, use_container_width=True)

    # Display Horizontal Bar Chart in a separate row
    st.plotly_chart(bar_fig, use_container_width=True)



# Sidebar for Team Analysis user inputs
st.sidebar.write("- - -")
st.sidebar.header("Team Analysis")
# Categorical multiselect
cat_filter = st.sidebar.multiselect("Compare Teams", df['Squad'].unique(), default=None)
# Function to create a slider in the sidebar
def create_slider(label, column_name):
    min_val = int(df[column_name].min())
    max_val = int(df[column_name].max())
    return st.sidebar.slider(f"Select Range for {label}", min_val, max_val, (min_val, max_val))

# Creating sliders for each specified metric
w_range = create_slider('Wins (W)', 'W')
sot_range = create_slider('Shots on Target', 'Shots_OT')
aw_range = create_slider('Aerials Won', 'Aerials_Won')
clr_range = create_slider('Clearances', 'Clearances')
blk_range = create_slider('Blocks', 'Blocks')

# Applying filters to the dataset
filtered_data = df
filtered_data = filtered_data[(filtered_data['W'] >= w_range[0]) & (filtered_data['W'] <= w_range[1])]
filtered_data = filtered_data[(filtered_data['Shots_OT'] >= sot_range[0]) & (filtered_data['Shots_OT'] <= sot_range[1])]
filtered_data = filtered_data[(filtered_data['Aerials_Won'] >= aw_range[0]) & (filtered_data['Aerials_Won'] <= aw_range[1])]
filtered_data = filtered_data[(filtered_data['Clearances'] >= clr_range[0]) & (filtered_data['Clearances'] <= clr_range[1])]
filtered_data = filtered_data[(filtered_data['Blocks'] >= blk_range[0]) & (filtered_data['Blocks'] <= blk_range[1])]
# ...
if cat_filter:
    filtered_data = filtered_data[filtered_data['Squad'].isin(cat_filter)]
# Display the filtered data
if st.sidebar.checkbox('Show Filtered Data'):
    st.write("- - -")
    st.subheader('Interactive Data')
    st.write(filtered_data)
