def save_data(df, output_path):
    """
    Save the processed DataFrame to a CSV file.
    """
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")