import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Cache the data loading function for speed.
@st.cache_data
def load_data():
    conn = sqlite3.connect("spotify.db")
    df = pd.read_sql_query("SELECT * FROM tracks WHERE script_deleted IS NULL", conn)
    conn.close()
    return df

def normalize_df(df, columns):
    """Apply min–max normalization on the specified columns."""
    df_norm = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df_norm[col]):
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if min_val != max_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
            else:
                df_norm[col] = 0.5  # if no variance, set a neutral value
    return df_norm

# Load data from the database.
df = load_data()

st.title("Enhanced Spotify Playlist Dashboard")
st.markdown("This interactive dashboard displays your Spotify tracks and playlists. Use the sidebar filters to adjust the data.")

# Sidebar Filters
st.sidebar.header("Filters")
playlist_options = sorted(df["playlist_name"].unique())
selected_playlists = st.sidebar.multiselect("Select Playlists", options=playlist_options, default=playlist_options)
pop_min = int(df["popularity"].min())
pop_max = int(df["popularity"].max())
pop_range = st.sidebar.slider("Popularity Range", min_value=pop_min, max_value=pop_max, value=(pop_min, pop_max))
if "release_year" in df.columns and not df["release_year"].isnull().all():
    year_min = int(df["release_year"].min())
    year_max = int(df["release_year"].max())
    selected_years = st.sidebar.slider("Release Year Range", min_value=year_min, max_value=year_max, value=(year_min, year_max))
else:
    selected_years = None

# Checkbox to enable normalization of numeric features.
normalize_data = st.sidebar.checkbox("Normalize Data for Visualization", value=True)

# Multi-select for advanced analysis: choose which audio features to include.
audio_features = [
    "danceability", "energy", "speechiness", "loudness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]
# Determine available features from the data.
available_features = [feat for feat in audio_features if feat in df.columns and df[feat].notnull().any()]
selected_advanced_features = st.sidebar.multiselect(
    "Select Audio Features for Advanced Analysis",
    options=available_features,
    default=available_features,
    help="Filter which attributes to display in advanced graphs. You can choose a subset if some attributes are dominating."
)

# Filter the data.
filtered_df = df[df["playlist_name"].isin(selected_playlists)]
filtered_df = filtered_df[(filtered_df["popularity"] >= pop_range[0]) & (filtered_df["popularity"] <= pop_range[1])]
if selected_years:
    filtered_df = filtered_df[(filtered_df["release_year"] >= selected_years[0]) & (filtered_df["release_year"] <= selected_years[1])]

st.markdown(f"Displaying **{len(filtered_df)}** tracks after filtering.")

# If normalization is enabled and features are available, create a normalized DataFrame.
if normalize_data and selected_advanced_features:
    filtered_norm = normalize_df(filtered_df, selected_advanced_features)
else:
    filtered_norm = filtered_df.copy()

# Basic Charts Section
st.subheader("Basic Charts")
if "energy" in filtered_df.columns:
    scatter_fig = px.scatter(
        filtered_df, x="energy", y="popularity", color="playlist_name",
        hover_data=["track_name", "release_year"], title="Popularity vs Energy"
    )
    st.plotly_chart(scatter_fig, use_container_width=True)
else:
    st.info("Energy data not available.")

if "release_year" in filtered_df.columns:
    hist_fig = px.histogram(filtered_df, x="release_year", nbins=20, title="Release Year Distribution")
    st.plotly_chart(hist_fig, use_container_width=True)
else:
    st.info("Release year data not available.")

count_df = filtered_df.groupby("playlist_name").size().reset_index(name="count")
bar_fig = px.bar(
    count_df, x="playlist_name", y="count",
    title="Track Count per Playlist",
    labels={"playlist_name": "Playlist", "count": "Number of Tracks"}
)
st.plotly_chart(bar_fig, use_container_width=True)

# Audio Feature Analysis Section
st.subheader("Audio Feature Analysis")
if selected_advanced_features:
    feature_choice = st.selectbox("Select an Audio Feature", options=selected_advanced_features)
    if normalize_data:
        feature_fig = px.histogram(filtered_norm, x=feature_choice, nbins=30, title=f"Distribution of Normalized {feature_choice.capitalize()}")
        avg_val = filtered_norm[feature_choice].mean()
    else:
        feature_fig = px.histogram(filtered_df, x=feature_choice, nbins=30, title=f"Distribution of {feature_choice.capitalize()}")
        avg_val = filtered_df[feature_choice].mean()
    st.plotly_chart(feature_fig, use_container_width=True)
    st.markdown(f"**Average {feature_choice.capitalize()}:** {avg_val:.2f}")
else:
    st.info("No audio feature data available.")

# Advanced Analysis Section with Tabs
st.subheader("Advanced Analysis")
tabs = st.tabs(["Correlation Heatmap", "Radar Chart", "Parallel Coordinates"])

with tabs[0]:
    st.markdown("### Correlation Heatmap")
    if selected_advanced_features:
        # Use normalized data if enabled.
        data_for_corr = filtered_norm[selected_advanced_features] if normalize_data else filtered_df[selected_advanced_features]
        corr = data_for_corr.corr()
        heatmap_fig = px.imshow(corr, text_auto=True, title="Correlation Matrix of Audio Features")
        st.plotly_chart(heatmap_fig, use_container_width=True)
    else:
        st.info("No audio feature data available for correlation analysis.")

with tabs[1]:
    st.markdown("### Radar Chart")
    if selected_advanced_features:
        # Compute average values per playlist from normalized or raw data.
        avg_df = (
            filtered_norm.groupby("playlist_name")[selected_advanced_features].mean().reset_index()
            if normalize_data
            else filtered_df.groupby("playlist_name")[selected_advanced_features].mean().reset_index()
        )
        categories = selected_advanced_features + [selected_advanced_features[0]]
        radar_fig = go.Figure()
        for _, row in avg_df.iterrows():
            values = [row[feat] for feat in selected_advanced_features]
            values += values[:1]  # close the loop
            radar_fig.add_trace(go.Scatterpolar(
                r=values, theta=categories, fill='toself', name=row["playlist_name"]
            ))
        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
            title="Average Audio Feature Profiles by Playlist"
        )
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.info("No audio feature data available for radar chart.")

with tabs[2]:
    st.markdown("### Parallel Coordinates Plot")
    if selected_advanced_features:
        avg_df = (
            filtered_norm.groupby("playlist_name")[selected_advanced_features].mean().reset_index()
            if normalize_data
            else filtered_df.groupby("playlist_name")[selected_advanced_features].mean().reset_index()
        )
        # Convert playlist names into numeric codes for the color property.
        avg_df["playlist_code"] = avg_df["playlist_name"].astype("category").cat.codes
        playlist_mapping = dict(enumerate(avg_df["playlist_name"].astype("category").cat.categories))
        parallel_fig = px.parallel_coordinates(
            avg_df,
            color="playlist_code",
            dimensions=selected_advanced_features,
            title="Parallel Coordinates Plot of Average Audio Features",
            labels={"playlist_code": "Playlist"}
        )
        # Update the colorbar to display the original playlist names.
        parallel_fig.update_coloraxes(colorbar=dict(
            tickvals=list(playlist_mapping.keys()),
            ticktext=list(playlist_mapping.values())
        ))
        st.plotly_chart(parallel_fig, use_container_width=True)
    else:
        st.info("No audio feature data available for parallel coordinates plot.")

# Option to view the underlying data table.
if st.checkbox("Show Data Table"):
    st.dataframe(filtered_df)

# Option to download filtered data as CSV.
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download filtered data as CSV",
    data=csv,
    file_name='filtered_spotify_tracks.csv',
    mime='text/csv',
)
