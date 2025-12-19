import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

st.sidebar.title("Parameter Interface")
feature_list = df.columns.tolist()
x_feature = st.sidebar.selectbox("Select X-axis feature", feature_list)
y_feature = st.sidebar.selectbox("Select Y-axis feature", feature_list)

k = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
cluster_choice = st.sidebar.selectbox("Select cluster to view", list(range(k)))

kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(df)
df["cluster"] = labels

st.title(" Iris K-Means Clustering Dashboard")
col_table, col_chart = st.columns(2)

with col_table:
    st.subheader(" Cluster Data Table")
    st.write(df[df["cluster"] == cluster_choice])

    if st.checkbox("Show full dataset"):
        st.write(df)


with col_chart:
    st.subheader(" Cluster Visualization")

    fig, ax = plt.subplots()
    for c in range(k):
        c_data = df[df["cluster"] == c]
        ax.scatter(
            c_data[x_feature],
            c_data[y_feature],
            label=f"Cluster {c}"
        )


    centroids = kmeans.cluster_centers_
    xi = feature_list.index(x_feature)
    yi = feature_list.index(y_feature)

    ax.scatter(
        centroids[:, xi], centroids[:, yi],
        marker="X", s=200, color="black", label="Centroids"
    )

    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.legend()

    st.pyplot(fig)

st.markdown("---")
st.header(" Predict Cluster for New Iris Instance")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
with col2:
    sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
with col3:
    petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)
with col4:
    petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)


if st.button("Predict Cluster"):
    new_point = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred_cluster = kmeans.predict(new_point)[0]

    st.success(f"This new iris instance belongs to **Cluster {pred_cluster}**")
