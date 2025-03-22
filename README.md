# Iris Dataset Analysis Project

## Project Overview
This project conducts an exploratory data analysis (EDA) on the Iris dataset, a widely recognized dataset in the field of machine learning. The goal is to investigate the relationships between the features—sepal length, sepal width, petal length, and petal width—and the three species of iris flowers: *Iris setosa*, *Iris versicolor*, and *Iris virginica*. The analysis involves data loading, cleaning, statistical summaries, and visualizations, with results presented in both a static report and an interactive Streamlit application.

## Dataset
The Iris dataset is provided in this repository as `iris.csv` within the `data/` folder. It comprises 150 samples, evenly distributed across the three species, with four numerical features per sample:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The dataset contains no missing values and is also accessible programmatically via `scikit-learn` using the `load_iris()` function.

## Project Structure
The repository is organized as follows:

iris_analysis_project/ ├── data/ │ └── iris.csv ├── notebooks/ │ └── exploratory_analysis.ipynb ├── reports/ │ ├── iris_eda_report.md │ └── iris_eda_streamlit.py ├── README.md └── requirement.txt


### File Descriptions
- **`data/iris.csv`**: The raw Iris dataset in CSV format, including column headers for straightforward parsing.
- **`notebooks/exploratory_analysis.ipynb`**: A Jupyter notebook detailing the complete EDA workflow, including data loading, cleaning, descriptive statistics, and visualizations such as scatter plots, histograms, and box plots.
- **`reports/iris_eda_report.md`**: A markdown file summarizing the EDA findings, featuring key insights, statistical highlights, and embedded visualization descriptions.
- **`reports/iris_eda_streamlit.py`**: A Python script for a Streamlit app that provides an interactive interface to explore the dataset, including dynamic visualizations and explanations of the findings.
- **`README.md`**: This file, offering an overview of the project, setup instructions, and usage details.

## Setup Instructions
To set up and run the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/iris_analysis_project.git
   cd iris_analysis_project

## How to Run
1. **Prerequisites**: Install Python and the required libraries (`pandas`, `seaborn`, `matplotlib`, `jupyter`). You can use:  
   ```bash
   pip install pandas seaborn matplotlib jupyter

# Setup and Execution Instructions

## Install Dependencies

Ensure Python 3.7+ is installed on your system.

Install the required libraries using pip:

```bash
   pip install pandas seaborn matplotlib scikit-learn streamlit

## Run the Jupyter Notebook 
'''bash 
   jupyter notebook eda_analysis.ipynb

## Run the Streamlit App
   streamlit run reports/iris_eda_streamlit.py

## Analysis Findings

The exploratory data analysis yielded the following insights:

## Species Differences

- **Iris setosa:**  
  Characterized by smaller petals (average length ~1.46 cm, width ~0.25 cm) and wider sepals.

- **Iris virginica:**  
  Exhibits the largest petals (average length ~5.55 cm, width ~2.03 cm) and longest sepals.

- **Iris versicolor:**  
  Shows intermediate measurements across features.

## Discriminative Features

- **Petal Measurements:**  
  Petal length and petal width demonstrate clear separation between species, making them highly effective for classification tasks.

- **Sepal Measurements:**  
  Sepal length and width show more overlap—particularly between versicolor and virginica—reducing their discriminative power.

## Distributions

- **Sepal Length:**  
  Displays a multimodal distribution, reflecting distinct species characteristics.

- **Petal Measurements:**  
  Tend to cluster by species, with setosa being notably distinct.

## Visualization Insights

- **Scatter Plots:**  
  Scatter plots of petal length vs. petal width reveal tight clustering for setosa and partial overlap between versicolor and virginica.

- **Box Plots:**  
  Emphasize the variability in petal sizes across species, with virginica showing the widest range.

These findings underscore the Iris dataset’s value as a benchmark for classification algorithms, particularly due to the strong discriminative power of petal measurements.

---

# How to Use the Streamlit App

The Streamlit app (`reports/iris_eda_streamlit.py`) offers an interactive experience for exploring the EDA results. Key features include:

- **Introduction:**  
  An overview of the Iris dataset and its importance in machine learning.

- **Dataset Overview:**  
  Displays the first few rows and structure of the dataset.

- **Summary Statistics:**  
  Provides mean, median, and standard deviation for each feature.

- **Species Comparison:**  
  Presents average feature values grouped by species.

- **Visualizations:**  
  Includes four interactive plot types:
  - **Line Chart:** Trends across features.
  - **Bar Chart:** Mean feature values by species.
  - **Histogram:** Feature distributions.
  - **Scatter Plot:** Relationships between pairs of features.

- **Key Insights:**  
  Summarizes the main findings and their implications.

To launch the app, run:

```bash
streamlit run reports/iris_eda_streamlit.py

# Observations

**Project Design:**  
The repository is structured for clarity and reproducibility, with distinct folders for data, analysis, and reports.

**Analysis Depth:**  
The Jupyter notebook provides a comprehensive, reproducible workflow, while the Streamlit app enhances accessibility with interactivity.

**Visualization Quality:**  
Plots are enhanced with titles, labels, and legends, ensuring clarity and interpretability.

**Robustness:**  
Error handling in the notebook ensures smooth execution, even with potential data loading issues.

**Educational Value:**  
This project serves as both a practical demonstration of EDA techniques and a resource for learning data analysis and visualization.

---

This project combines technical rigor with user-friendly presentation, making it an excellent example for data science enthusiasts and practitioners alike. For any questions or contributions, feel free to open an issue or pull request on the repository!
