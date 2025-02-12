import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def univariate_analysis(df, max_plots=5):
    print("Univariate Analysis")
    
    # Summary statistics
    print("Summary Statistics:")
    print(df.describe())
    
    # Histograms for numeric columns
    numeric_columns = df.select_dtypes(include=[float, int]).columns
 
    
    
    # Boxplots for numeric columns
    if len(numeric_columns) == 0:
        print("No numeric columns found for boxplots.")
    else:
        for i, column in enumerate(numeric_columns):
            if i >= max_plots:
                break
            plt.figure(figsize=(10, 5))
            sns.boxplot(x=df[column])
            plt.title(f'Boxplot of {column}')
            plt.xlabel(column)
            plt.show()
    
    # Count plots for categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) == 0:
        print("No categorical columns found for count plots.")
    else:
        for i, column in enumerate(categorical_columns):
            if i >= max_plots:
                break
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
    if len(numeric_df.columns) == 0:
        print("No numeric columns found for correlation matrix.")
    else:
        plt.figure(figsize=(12, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
        
        # Pairplot for numeric columns
        sns.pairplot(numeric_df, plot_kws={'alpha': 0.5})
        plt.suptitle('Pairplot', y=1.02)
        plt.show()
        
        # Scatter plots for numeric columns
        for x_column in numeric_df.columns:
            for y_column in numeric_df.columns:
                if x_column != y_column:
                    plt.figure(figsize=(10, 5))
                    sns.scatterplot(x=numeric_df[x_column], y=numeric_df[y_column])
                    plt.title(f'Scatterplot of {x_column} vs {y_column}')
                    plt.xlabel(x_column)
                    plt.ylabel(y_column)
                    plt.show()

def exploratory_data_analysis(df, title='EDA'):
    print(f'Exploratory Data Analysis for {title}')
    
    # Univariate Analysis
    univariate_analysis(df)
    
    # Bivariate Analysis
    bivariate_analysis(df)