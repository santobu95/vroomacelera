#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Geofence Clustering App with Outlier Filtering
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium
from shapely.geometry import MultiPoint

st.set_page_config(page_title="Geofence Clustering", layout="centered")
st.title("📍 Geofence Clustering App")

# Predefined named colors for clusters
named_colors = [
    "red", "blue", "green", "orange", "purple",
    "cyan", "magenta", "brown", "pink", "gray"
]

# Upload file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df = df.dropna(subset=['lat', 'lng', 'completion_rate'])

    st.success("File uploaded and cleaned!")

    # Select features for clustering
    clustering_cols = st.multiselect(
        "Select features to use for clustering :) | If first try not satisfying try removing completion rate",
        options=df.columns.tolist(),
        default=["lat", "lng", "orders_initial", "completion_rate"]
    )

    if clustering_cols:
        # Normalize and cluster
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(df[clustering_cols])

        n_clusters = st.slider("Select number of clusters | Recommended amount is 4", 2, 10, 4)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(scaled_features)

        # Assign readable color names
        cluster_colors = {
            i: named_colors[i % len(named_colors)] for i in range(n_clusters)
        }

        # Show cluster summary
        st.subheader("Cluster Summary")
        summary = df.groupby('cluster').agg({
            'completion_rate': 'mean',
            'orders_initial': 'mean'
        }).reset_index()

        summary['color'] = summary['cluster'].map(cluster_colors)
        summary.rename(columns={
            'completion_rate': 'avg_completion_rate',
            'orders_initial': 'avg_orders_initial'
        }, inplace=True)

        st.dataframe(summary)

        # Show polygon coordinates per cluster with outlier removal
        st.subheader("Cluster Polygon Coordinates (Copy and Paste them in Duse-Eye)")
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_df = df[df['cluster'] == cluster_id][['lat', 'lng']].copy()

            # Calculate centroid
            centroid = cluster_df[['lat', 'lng']].mean().values

            # Calculate Euclidean distance from centroid
            cluster_df['dist'] = np.sqrt((cluster_df['lat'] - centroid[0])**2 + (cluster_df['lng'] - centroid[1])**2)

            # Remove top 5% farthest points (outliers)
            filtered_cluster = cluster_df[cluster_df['dist'] <= cluster_df['dist'].quantile(0.98)]

            # Continue only if we have enough points
            if len(filtered_cluster) < 3:
                st.warning(f"Cluster {cluster_id} doesn't have enough non-outlier points for a polygon.")
                continue

            cluster_points = filtered_cluster[['lng', 'lat']].values
            hull = MultiPoint(cluster_points).convex_hull
            coords = list(hull.exterior.coords) if hull.geom_type == 'Polygon' else list(hull.coords)
            coord_text = "\n".join([f"{lng:.6f}, {lat:.6f}" for lng, lat in coords])

            with st.expander(f"Cluster {cluster_id} ({cluster_colors[cluster_id]}) Polygon Coordinates"):
                st.text_area("Copy-paste friendly coordinates (lat, lng):", coord_text, height=200)

        # Visualize map
        st.subheader("Cluster Map")
        m = folium.Map(location=[df['lat'].mean(), df['lng'].mean()], zoom_start=12)
        for _, row in df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lng']],
                radius=4,
                color=cluster_colors[row['cluster']],
                fill=True,
                fill_color=cluster_colors[row['cluster']],
                fill_opacity=0.3,
                opacity=0.3,
                popup=f"CR: {row['completion_rate']:.2f}, Cluster: {row['cluster']}"
            ).add_to(m)

        st_folium(m, width=700, height=500)

