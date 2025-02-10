import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def univariate_analysis(df):
    print("Univariate Analysis")
    
    # Summary statistics
    print("Summary Statistics:")
    print(df.describe())
    
    # Histograms for numeric columns
    numeric_columns = df.select_dtypes(include=[float, int]).columns
    for column in numeric_columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[column], kde=True, bins=30)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
    
    # Boxplots for numeric columns
    for column in numeric_columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=df[column])
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.show()
    
    # Count plots for categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(y=df[column])
        plt.title(f'Countplot of {column}')
        plt.xlabel('Count')
        plt.ylabel(column)
        plt.show()

def bivariate_analysis(df):
    print("Bivariate Analysis")
    
    # Correlation matrix
    numeric_df = df.select_dtypes(include=[float, int])
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
    
    # Pairplot for numeric columns
    sns.pairplot(numeric_df, plot_kws={'alpha': 0.5})
    plt.suptitle('Pairplot', y=1.02)
    plt.show()
    
    # Scatter plots for numeric columns
    for column in numeric_df.columns:
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=numeric_df.index, y=numeric_df[column])
        plt.title(f'Scatterplot of {column}')
        plt.xlabel('Index')
        plt.ylabel(column)
        plt.show()

def exploratory_data_analysis(df, title='EDA'):
    print(f'Exploratory Data Analysis for {title}')
    
    # Univariate Analysis
    univariate_analysis(df)
    
    # Bivariate Analysis
    bivariate_analysis(df)