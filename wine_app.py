# import libraries

import streamlit as st 
import pickle
import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# load wine dataset
df = pd.read_csv('wine_deployed_data.csv', index_col=0)
col = df.columns.tolist()

# scaling df
se = StandardScaler()
df_scaled = se.fit_transform(df)

# load pickle file
def load_model():
    with open('kmean.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


# sidebar interface
st.sidebar.title('Wine Anlayst Toolbox 🧰')
with st.sidebar:
    st.header("Project Details")
    st.markdown('**Name:** Khant Razar Kyaw')
    st.markdown('**Course:** Introduction Machine Learning')
    st.markdown('**Application:** Wine Clustering Application')
    
    st.divider()
    
    st.header("🛠️ How to Use")
    st.markdown("""
    1. **Explore Clusters:** - Use the *top section* to test different numbers of clusters (K).
       - Check the **Elbow** and **Silhouette** graphs to find the best K.
    2. **Visualize:** - Use the scatter plot at the middle to see how wine components make the wine cluster.
    3. **Input Wine Data:** - Adjust the sliders in the *bottom section* to match your wine's chemical properties.
    4. **Predict:** - Click **"Let's Group Wine Category"** to see which group your wine belongs to. 
    """)
    
    st.divider()
    
    st.header("📖 Data Dictionary")
    st.caption("Expand below to understand the chemical features.")
    
    with st.expander("🧪 1. Composition (Acidity & Body)"):
        st.markdown("""
        * **Alcohol:** The percent alcohol by volume. Gives wine its "body" and warmth.
        * **Malic Acid:** The primary acid in grapes (tastes like green apples). High levels = tart wine.
        * **Ash:** The residue left after evaporation. Indicates total mineral content.
        * **Alcalinity of Ash:** A measure of the basicity of the ash. Affects the wine's pH stability.
        """)

    with st.expander("🪨 2. Minerals & Nutrients"):
        st.markdown("""
        * **Magnesium:** An essential nutrient for yeast during fermentation.
        * **Proline:** The most abundant amino acid. Indicates grape maturity/ripeness.
        """)

    with st.expander("🍇 3. Phenolics (Flavor & Tannin)"):
        st.markdown("""
        * **Total Phenols:** The sum of all phenolic compounds (tannins, color, flavors).
        * **Flavanoids:** The main antioxidants in wine. They contribute to bitterness and mouthfeel.
        * **Nonflavanoid Phenols:** Smaller acids (like caftaric acid). Too much can cause browning.
        * **Proanthocyanins:** Also known as condensed tannins. They give the "drying" sensation in red wines.
        """)

    with st.expander("🎨 4. Visuals & Quality"):
        st.markdown("""
        * **Color Intensity:** How dark or deep the wine color is.
        * **Hue:** The shade of the color (Purple = Young, Orange/Brick = Old).
        * **OD280/OD315:** A lab test measuring protein purity. Higher values often indicate better authenticity and quality.
        """)
        
    st.markdown("---")

    st.caption("Developed for Machine Learning Final Project at Parami University")


# main interface
st.title('Wine Clustering Dashboard 🍷')

st.markdown("""
Welcome to the **Wine Clustering Dashboard Application**. 
This tool uses Unsupervised Machine Learning algorithms to analyze the chemical composition of wines and 
group them into distinct "clusters" (types) without human intervention.

**🎯 Application Objective:**
To identify natural groupings among wine samples using 13 chemical features 
(such as *Alcohol*, *Flavanoids*, and *Color Intensity*), 
helping winemakers characterize wine profiles automatically.
""")

st.markdown("---")
col_1, col_2 = st.columns(2)

with col_1:
    st.header("Choosing the Right Number of Clusters")
    st.write('In the clustering of wine using unsupervised learning, ' \
    'we can anticipate different wine groups based on our preference.')
    st.write('The **Evaluation Analysis** measures how good the groups ' \
    'are clustered properly.')
    st.write('There are two essential evaluation analysis for unsupervised learning, the Elbow Method and Silhouette Score.')
    st.write('**Elbow Method or WCSS Score** measures the compactness of ' \
    'the clusters (which is the lower the better).')
    st.write('**The Silhouette Score**, on the other hand, measures the separation ' \
    'between clusters (which is the greater the better).')
    k3 = st.slider("Number of Wine Groups", 2, 15, 3)
    
    # elbow method
    inertia = []
    kmean3 = KMeans(n_clusters = k3, random_state = 25)
    kmean3.fit(df_scaled)
    wcss3 = kmean3.inertia_
    sl_score3 = silhouette_score(df_scaled, kmean3.labels_)
    if st.button("Clustering Analysis"):
        st.success(f"For the Wine Cluster of **{k3}**, the WCSS value is **{round(wcss3, 2)}** and the Silhouette Score is **{round(sl_score3, 2)}**.")
    
    st.write('Here, based on how bend the elbow angle of WCSS scores or how large is Silhouette Score, we can decide which is the best number of clusters for our model.')

with col_2:
    st.header("Clustering Evaluation Graphs")
    st.write("The clustering graphs of the **Elbow Method (WCSS Score)** and the **Silhouetter Score** " \
    "show how good a clustering model is based on the number of clusters.")
    st.write("Let's choose number of clusters, and visualize how those scores changed with number of clusters.")

    k4 = st.slider("Number of Wine Clusters", 2, 15, 3)
    wcss4 = []
    sl_score4 = []
    for x4 in range(2, k4+1):
        kmean4 = KMeans(n_clusters = x4, random_state = 17)
        kmean4.fit(df_scaled)
        wcss4.append(kmean4.inertia_)
        sl_score4.append(silhouette_score(df_scaled, kmean4.labels_))
    
    st.write("Based on what you chose number of wine grouping, we will visualize the Elbow Method Score and Silhouette Score separately.")
    tab1, tab2 = st.tabs(["📉 Elbow Method", "📊 Silhouette Score"])

    with tab1:
        st.subheader("Elbow Method (Inertia)")
        fig_1, ax_1 = plt.subplots()
        sns.lineplot(x = range(2, k4+1), y = wcss4, marker='X', color='blue', ax=ax_1)
        ax_1.set_xlabel("Number of Clusters (k)")
        ax_1.set_ylabel("WCSS")
        st.pyplot(fig_1)
        st.caption("Look for the 'bend' (elbow) where the score stops dropping fast.")

    with tab2:
        st.subheader("Silhouette Score")
        fig_2, ax_2 = plt.subplots()
        sns.lineplot(x = range(2, k4+1), y = sl_score4, marker='o', color='red', ax=ax_2)
        ax_2.set_xlabel("Number of Clusters (k)")
        ax_2.set_ylabel("Silhouette Coefficient")
        st.pyplot(fig_2)
        st.caption("Higher is better. Closer to +1 means well-separated clusters.")
st.markdown("---") 

st.write("After analyzing and visualization the number of optimal clusters for Wine Clustering, " \
"let's annotate how wine components makes cluster.")

st.subheader('Cluster Dispersion based on Wine Components')
vizx = st.selectbox("Choose X-Axis", col)
vizy = st.selectbox("Choose Y-Axis", col)

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(
    data=df, x=vizx, y=vizy, hue = 'wine_cluster',
    palette='viridis', alpha = 0.9, s=100, ax=ax
    )
st.pyplot(fig)

st.markdown('---')

st.header("Let's Group Wine Together!!🍷")
col_3, col_4 = st.columns(2)

with col_3:
    st.subheader('Alcohol, Ash & Acidity')
    # 1. Alcohol, Ash & Acidity
    with st.expander("🍷 Alcohol, Ash & Acidity (Body & Tartness)"):
        st.markdown("""
    **1. Alcohol:**
    * Comes from yeast fermenting the sugar in grapes.
    
    **2. Malic Acid:**
    * Winemakers often use "Malolactic Fermentation" to turn sharp *Malic Acid* into creamy *Lactic Acid* (like in buttery Chardonnays).
    
    **3. Ash & Alkalinity:**
    * Affects how "fresh" or "flat" the wine tastes.
        """)
    s1 = st.slider("Alcohol Content", float(df.Alcohol.min()), float(df.Alcohol.max()), value=float(df.Alcohol.mean()))
    s2 = st.slider("Malic Acid", float(df.Malic_Acid.min()), float(df.Malic_Acid.max()), value=float(df.Malic_Acid.mean()))
    s3 = st.slider("Ash",  float(df.Ash.min()), float(df.Ash.max()),  value=float(df.Ash.mean()))
    s4 = st.slider("Alcalinity of Ash", float(df.Ash_Alcanity.min()), float(df.Ash_Alcanity.max()),  value=float(df.Ash_Alcanity.mean()))
    
    st.subheader('Minerals')
    # 2. Minerals
    with st.expander("🪨 Minerals (Yeast Health & Mouthfeel)"):
        st.markdown("""
    **1. Magnesium:**
    * Healthy yeast needs magnesium to finish fermentation without producing bad smells (like rotten eggs).
    
    **2. Proline:**
    * Indicates grape ripeness. Unlike other amino acids, yeast doesn't eat Proline much, so it stays in the wine and adds a subtle sweetness and viscosity (thickness).
        """)
    s5 = st.slider("Magnesium (mg)",  int(df.Magnesium.min()), int(df.Magnesium.max()),  step=1, value=int(df.Magnesium.mean()))
    s13 = st.slider("Proline",  int(df.Proline.min()), int(df.Proline.max()),   step=10, value=int(df.Proline.mean()))

with col_4:
    st.subheader('Phenolics & Flavanoids')
    # 3. Phenolics (Flavor & Bitterness)
    with st.expander("🍇 Phenolics (Taste, Bitterness & Aging)"):
        st.markdown("""
    **1. Total Phenols & Flavanoids:**
    * High phenols = more aging potential but can be bitter if not balanced.
    
    **2. Nonflavanoid Phenols:**
    * Too much can lead to browning (oxidation).
    
    **3. Proanthocyanins:**
    * *They give the "drying" sensation in red wine. Essential for red wine texture.
        """)
    s6 = st.slider("Total Phenols", float(df.Total_Phenols.min()), float(df.Total_Phenols.max()), value=float(df.Total_Phenols.mean()))
    s7 = st.slider("Flavanoids", float(df.Flavanoids.min()), float(df.Flavanoids.max()), value=float(df.Flavanoids.mean()))
    s8 = st.slider("Nonflavanoid Phenols", float(df.Nonflavanoid_Phenols.min()), float(df.Nonflavanoid_Phenols.max()),  step=0.01, value=float(df.Nonflavanoid_Phenols.mean()))
    s9 = st.slider("Proanthocyanins", float(df.Proanthocyanins.min()), float(df.Proanthocyanins.max()),  value=float(df.Proanthocyanins.mean()))
    
    st.subheader('Visual & Quality Metrics')
    # 4. Color & Clarity
    with st.expander("🎨 Color & Quality (Visuals)"):
        st.markdown("""
    **1. Color Intensity:**
    * Darker usually means thicker skin grapes and more extraction during fermentation.
    
    **2. Hue:**
    * Young red wines are purple/ruby. Old wines turn "brick" or orange-brown.
    
    **3. OD280 (Protein Content):**
    * Used to detect authenticity or dilution. High values often mean better quality/purity.
        """)
    
    
    s10 = st.slider("Color Intensity", float(df.Color_Intensity.min()), float(df.Color_Intensity.max()), value=float(df.Color_Intensity.mean()))
    s11 = st.slider("Hue", float(df.Hue.min()), float(df.Hue.max()), value=float(df.Hue.mean()))
    s12 = st.slider("OD280 of Diluted Wines",  float(df.OD280.min()), float(df.OD280.max()),  value=float(df.OD280.mean()))
    


st.markdown('---')
col11, col12, col13= st.columns(3)

with col12:
    if st.button("Let's Group Wine Category"):
        model = load_model()
        input_df = pd.DataFrame({
            'alcohol': [s1],
            'malic_acid': [s2],
            'ash': [s3],
            'alcalinity_of_ash': [s4],
            'magnesium': [s5],
            'total_phenols': [s6],
            'flavanoids': [s7],
            'nonflavanoid_phenols': [s8],
            'proanthocyanins': [s9],
            'color_intensity': [s10],
            'hue': [s11],
            'od280/od315_of_diluted_wines': [s12],
            'proline': [s13]
        })
        input_df_scaled = se.fit_transform(input_df)
        result = model.predict(input_df_scaled)
        target_names = ['Wine_Category_1', 'Wine_Category_2', 'Wine_Category_3']
        input_df['wine_cluster'] = result
        final_result = target_names[result[0]]
        st.success(f"The Wine Category is {final_result}")

st.markdown('---')

st.write('**How was your experience with the wine clustering analysis?**')
st.write('Your feedback helps us improve the model accuracy and app usability.')

rating = st.slider("Rate this App (1 = Poor, 5 = Excellent)", 1, 5, 5)
if rating >= 4:
    st.write("🌟🌟🌟🌟🌟 Thank you! We are glad you liked it.")
elif rating >= 2:
    st.write("⭐⭐⭐ Thanks for your feedback!")
else:
    st.write("⭐ Sorry for any user inconvinience.")

st.markdown('---')
col11, col12, col13= st.columns(3)

with col12:
   google_form_url = 'https://docs.google.com/forms/d/e/1FAIpQLScK_QIlup9f5E_I2scdssaf3aGOnqdNq_xaT3jaHjN0Rbu7ww/viewform?usp=header'
   st.link_button("📝 Please Take Short Survey", google_form_url)