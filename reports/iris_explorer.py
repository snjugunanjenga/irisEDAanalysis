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

# Title
st.title("Iris Dataset Explorer")

# Introduction
st.write("""
Welcome to the Iris Dataset Explorer! The Iris dataset is a classic in the world of machine learning, 
first introduced by the statistician Ronald Fisher in 1936. It contains measurements of 150 iris flowers 
from three different species: setosa, versicolor, and virginica. Each flower is described by four features: 
sepal length, sepal width, petal length, and petal width. In this app, we’ll explore the dataset to uncover 
interesting patterns and relationships between these features and the species.
""")

# Dataset Overview
st.header("Dataset Overview")
st.write("Let’s start by taking a look at the dataset. Here are the first five rows:")
st.dataframe(df.head())
st.write(f"The dataset consists of {df.shape[0]} samples, each with {df.shape[1]} columns: the four features and the species label. Fortunately, there are no missing values, so we can proceed directly to analysis.")

# Summary Statistics
st.header("Summary Statistics")
st.write("Now, let’s examine the summary statistics of the numerical features:")
st.table(df.describe())
st.write("""
From these statistics, we can see that the features have different scales. For example, sepal length ranges 
from 4.3 to 7.9 cm, while petal width ranges from 0.1 to 2.5 cm. The standard deviations also vary, 
indicating different levels of variability in the measurements.
""")

# Species Comparison
st.header("Species Comparison")
st.write("To understand how the species differ, let’s look at the average measurements for each species:")
grouped = df.groupby('species').mean()
st.table(grouped)
st.write("""
Here, we can see that setosa has the smallest petal length and width on average, while virginica has the largest. 
Interestingly, setosa has a larger sepal width compared to the other species. This suggests that petal measurements 
might be more useful for distinguishing between the species.
""")

# Visualizations
st.header("Visualizations")

# Pair Plot
st.subheader("Pair Plot")
st.write("One of the most informative visualizations is the pair plot, which shows the relationships between all pairs of features, colored by species:")
sns.pairplot(df, hue='species', diag_kind='hist')
st.pyplot(plt.gcf())
plt.clf()
st.write("""
In this plot, we can clearly see that setosa forms a distinct cluster, especially when looking at petal length 
and petal width. Notice how the blue points (setosa) form a tight cluster in the bottom-left of the petal length 
vs. petal width scatter plot, while versicolor and virginica overlap more.
""")

# Box Plots
st.subheader("Box Plots")
st.write("To further explore the distributions, let’s look at box plots for each feature across the species:")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, feature in enumerate(df.columns[:-1]):
    sns.boxplot(x='species', y=feature, data=df, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(feature)
plt.tight_layout()
st.pyplot(fig)
plt.clf()
st.write("""
These box plots confirm that petal length and width have less overlap between species compared to sepal measurements. 
For instance, the petal length of setosa is entirely below that of versicolor and virginica, making these features 
excellent for classification.
""")

# Correlation Matrix
st.subheader("Correlation Matrix")
st.write("Finally, let’s examine the correlations between the numerical features:")
corr = df.drop('species', axis=1).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
st.pyplot(plt.gcf())
plt.clf()
st.write("""
The correlation matrix shows a strong positive correlation between petal length and petal width (0.96), meaning 
that flowers with longer petals also tend to have wider petals. This strong relationship can be useful in understanding 
the underlying biology of the flowers or in feature selection for machine learning models.
""")

# Key Insights
st.header("Key Insights")
st.write("""
Based on our analysis, here are the key insights:

- The Iris dataset consists of three species with distinct characteristics, particularly in their petal measurements.
- Setosa is easily distinguishable from versicolor and virginica due to its smaller petals.
- Petal length and petal width are highly correlated and provide the most information for classifying the species.
- Sepal measurements, while still useful, are less effective for separating the species due to more overlap in their distributions.

These findings have important implications for machine learning. The clear separation of setosa suggests that a simple 
classifier, like a decision tree or k-nearest neighbors, could achieve high accuracy. The overlap between versicolor 
and virginica indicates that more sophisticated models or additional features might be needed to perfectly classify 
all samples. Moreover, the strong correlation between petal length and width means that we might not need both features 
for classification, potentially allowing for dimensionality reduction.
""")