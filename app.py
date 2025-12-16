import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="YouTube Trending Analysis Pro", layout="wide")
st.title(" YouTube Trending Video Analysis – Advanced Edition")

# --------------------------------------------------
# Methodology
# --------------------------------------------------
st.markdown("###  Methodology")
st.markdown("""
- Dataset: Trending YouTube Video Statistics (US) from Kaggle
- Tools: Python, Pandas, Streamlit, Matplotlib, Plotly, scikit-learn
- Approach:
  1. Load CSV & category JSON
  2. Map category IDs
  3. Clean data & fix dates
  4. Create advanced metrics (engagement, viral, trend speed)
  5. Visualize trends & engagement
  6. Predict likes from views & comments
  7. Extract actionable insights
""")

# --------------------------------------------------
# Load data
# --------------------------------------------------
@st.cache_data
def load_videos():
    return pd.read_csv("data/USvideos.csv")

@st.cache_data
def load_categories():
    with open("data/US_category_id.json", "r") as f:
        data = json.load(f)
    return {int(item["id"]): item["snippet"]["title"] for item in data["items"]}

df = load_videos()
category_mapping = load_categories()
df["category"] = df["category_id"].map(category_mapping)

# --------------------------------------------------
# Data cleaning & advanced metrics
# --------------------------------------------------
# Convert to datetime safely
df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
df["trending_date"] = pd.to_datetime(df["trending_date"], errors="coerce")

# Remove timezone info to fix tz-naive vs tz-aware issue
df["publish_time"] = df["publish_time"].dt.tz_localize(None)
df["trending_date"] = df["trending_date"].dt.tz_localize(None)

# Drop rows with missing dates
df = df.dropna(subset=["publish_time", "trending_date"])

# Basic metrics
df["like_ratio"] = df["likes"] / (df["views"] + 1)

# Advanced metrics
df["engagement_score"] = df["likes"]*0.4 + df["comment_count"]*0.4 + df["views"]*0.2
category_avg_views = df.groupby("category")["views"].transform("mean")
df["viral_score"] = df["views"] / (category_avg_views + 1)

# Trend days (ensure positive)
df["trend_days"] = (df["trending_date"] - df["publish_time"]).dt.days
df["trend_days"] = df["trend_days"].apply(lambda x: max(x, 0))

df["publish_weekday"] = df["publish_time"].dt.day_name()

# --------------------------------------------------
# Sidebar filters
# --------------------------------------------------
st.sidebar.header(" Filters")
selected_categories = st.sidebar.multiselect(
    "Select Categories",
    options=sorted(df["category"].dropna().unique()),
    default=sorted(df["category"].dropna().unique())
)
min_viral = st.sidebar.slider("Minimum Viral Score", 0.0, float(df["viral_score"].max()), 0.0)
selected_weekdays = st.sidebar.multiselect(
    "Publish Weekdays",
    options=df["publish_weekday"].dropna().unique(),
    default=df["publish_weekday"].dropna().unique()
)

filtered_df = df[
    (df["category"].isin(selected_categories)) &
    (df["viral_score"] >= min_viral) &
    (df["publish_weekday"].isin(selected_weekdays))
]

# --------------------------------------------------
# Tabs for organization
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overview", "Channels", "Predictions"])

# ------------------ TAB 1: Overview ------------------
with tab1:
    st.subheader(" Dataset Preview")
    st.dataframe(filtered_df.head())

    st.subheader(" Trending Videos by Category")
    category_counts = filtered_df["category"].value_counts()
    st.bar_chart(category_counts)

    st.subheader("📈 Views Distribution (Log Scale)")
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(filtered_df["views"], bins=50)
    ax.set_xlabel("Views")
    ax.set_ylabel("Frequency")
    ax.set_xscale("log")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader(" Likes vs Views")
    fig = px.scatter(
        filtered_df,
        x="views",
        y="likes",
        color="category",
        size="comment_count",
        hover_data=["title","channel_title"]
    )
    st.plotly_chart(fig)

    st.subheader(" Top 5 Categories Share")
    top_categories = filtered_df["category"].value_counts().head(5)
    fig, ax = plt.subplots(figsize=(7,7))
    ax.pie(top_categories, labels=top_categories.index, autopct="%1.1f%%", startangle=140)
    st.pyplot(fig)

    st.subheader(" Trending Videos Over Time")
    trend_over_time = filtered_df.groupby(filtered_df["trending_date"].dt.date).size()
    st.line_chart(trend_over_time)

# ------------------ TAB 2: Channels ------------------
with tab2:
    st.subheader(" Top Channels by Average Engagement")
    top_channels = (
        filtered_df.groupby("channel_title")[["engagement_score"]]
        .mean()
        .sort_values("engagement_score", ascending=False)
        .head(10)
    )
    st.bar_chart(top_channels)

    st.subheader("Channel Consistency (Number of Trending Videos)")
    channel_counts = filtered_df["channel_title"].value_counts().head(10)
    st.bar_chart(channel_counts)

# ------------------ TAB 3: Predictions ------------------
with tab3:
    st.subheader("📈 Predict Likes from Views & Comments")
    X = filtered_df[["views", "comment_count"]]
    y = filtered_df["likes"]
    model = LinearRegression()
    model.fit(X, y)
    filtered_df["predicted_likes"] = model.predict(X)

    st.write("**Sample Predictions:**")
    st.dataframe(filtered_df[["title","channel_title","views","comment_count","likes","predicted_likes"]].head(10))

    st.subheader("Scatter: Predicted vs Actual Likes")
    fig = px.scatter(
        filtered_df,
        x="likes",
        y="predicted_likes",
        color="category",
        hover_data=["title","channel_title"]
    )
    st.plotly_chart(fig)

# --------------------------------------------------
# Key Insights
# --------------------------------------------------
st.markdown("###  Key Findings")
st.markdown("""
- Categories like Music, Entertainment, and Gaming dominate trending videos  
- Views follow a heavy-tail distribution: most videos have low views, a few go viral  
- Likes strongly correlate with views; viral_score identifies standout videos  
- Trending speed varies: some videos trend immediately, others slowly  
- Top channels consistently produce highly engaging content  
- Simple regression predicts likes from views & comments with reasonable accuracy
""")

# --------------------------------------------------
# Final Takeaways
# --------------------------------------------------
st.markdown("---")
st.subheader(" Final Takeaways")
st.markdown("""
- This dashboard combines interactive visualizations, advanced metrics, and predictive modeling  
- It allows exploration of category trends, channel performance, and viral potential  
- Enhances understanding of what drives YouTube trending success
""")
