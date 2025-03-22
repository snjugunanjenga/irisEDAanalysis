import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Streamlit app layout
st.title("Exploratory Data Analysis of the Iris Dataset")

### Introduction
st.header("Introduction")
st.write("""
The Iris dataset is a classic dataset in machine learning, introduced by Ronald Fisher in 1936. 
It contains measurements of iris flowers from three species: *Iris setosa*, *Iris versicolor*, and *Iris virginica*. 
With 150 samples and four features (sepal length, sepal width, petal length, and petal width), this dataset is ideal 
for exploring relationships between floral characteristics and species. This app presents the findings from our EDA, 
including visualizations and insights.
""")

### Dataset Overview
st.header("Dataset Overview")
st.write("The dataset has **150 samples** across three species, with four numerical features and no missing values. Here are the first five rows:")
st.dataframe(df.head())
st.write(f"Dataset shape: {df.shape} (150 rows, 5 columns including species)")

### Summary Statistics
st.header("Summary Statistics")
st.write("Below are the summary statistics for the numerical features (in cm):")
st.write(df.describe())

### Species Comparison
st.header("Species Comparison")
st.write("Mean values of each feature grouped by species:")
grouped = df.groupby('species').mean()
st.write(grouped)

### Visualizations
st.header("Visualizations")
st.write("Explore the dataset through four visualizations: a line chart, bar chart, histogram, and scatter plot.")

#### Visualization 1: Line Chart
st.subheader("Line Chart: Sorted Petal Length by Species")
st.write("This plot shows petal lengths sorted in ascending order for each species.")
fig1, ax1 = plt.subplots(figsize=(10, 6))
for species in df['species'].unique():
    species_data = df[df['species'] == species].sort_values('petal length (cm)')
    ax1.plot(range(len(species_data)), species_data['petal length (cm)'], label=species)
ax1.set_title('Sorted Petal Length by Species')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Petal Length (cm)')
ax1.legend()
st.pyplot(fig1)
st.write("**Insight**: Setosa has smaller petal lengths (<2 cm), while virginica reaches up to ~6.9 cm.")

#### Visualization 2: Bar Chart
st.subheader("Bar Chart: Average Petal Length by Species")
st.write("This chart compares the average petal length across species.")
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.barplot(x='species', y='petal length (cm)', data=df, ax=ax2)
ax2.set_title('Average Petal Length by Species')
ax2.set_xlabel('Species')
ax2.set_ylabel('Petal Length (cm)')
st.pyplot(fig2)
st.write("**Insight**: Virginica has the longest average petal length (~5.5 cm), followed by versicolor (~4.3 cm), and setosa (~1.5 cm).")

#### Visualization 3: Histogram
st.subheader("Histogram: Distribution of Sepal Length")
st.write("This histogram shows the distribution of sepal lengths across all samples.")
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.histplot(df['sepal length (cm)'], bins=20, kde=True, ax=ax3)
ax3.set_title('Distribution of Sepal Length')
ax3.set_xlabel('Sepal Length (cm)')
ax3.set_ylabel('Frequency')
st.pyplot(fig3)
st.write("**Insight**: Sepal length is roughly normally distributed with peaks around 5, 6, and 6.5 cm, reflecting species differences.")

#### Visualization 4: Scatter Plot
st.subheader("Scatter Plot: Sepal Length vs. Petal Length by Species")
st.write("This plot shows the relationship between sepal length and petal length, colored by species.")
fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df, ax=ax4)
ax4.set_title('Sepal Length vs. Petal Length by Species')
ax4.set_xlabel('Sepal Length (cm)')
ax4.set_ylabel('Petal Length (cm)')
ax4.legend(title='Species')
st.pyplot(fig4)
st.write("**Insight**: Setosa forms a distinct cluster (petal length ~1-2 cm), while versicolor and virginica overlap but differ in petal length ranges.")

### Key Insights
st.header("Key Insights")
st.write("""
- **Dataset**: 150 samples, 3 species, 4 features, no missing values.
- **Species Differences**: 
  - Setosa: Small petals (~1.46 cm length, ~0.25 cm width).
  - Virginica: Largest petals (~5.55 cm length, ~2.03 cm width).
  - Versicolor: Moderate sizes.
- **Key Features**: Petal length and width are more discriminative than sepal measurements.
- **Utility**: The dataset is well-suited for classification, especially with petal features.
""")