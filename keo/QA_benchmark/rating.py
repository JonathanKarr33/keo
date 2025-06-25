import os
import pandas as pd

# Get the current directory
target_dir = os.getcwd()

# List all files in the directory with "Evaluation" in their name and .csv extension
file_name_list = [file for file in os.listdir(target_dir) if "Evaluation" in file and "GPT4o" in file and file.endswith(".csv")]

# Process each file
for file_name in file_name_list:
    file_path = os.path.join(target_dir, file_name)

    try:
        # Load the CSV file
        data = pd.read_csv(file_path)

        # Print the file name without the extension
        print(f"File: {file_name}")

        # Calculate the mean and standard deviation of the 'rating' column
        if 'rating' in data.columns:
            mean_rating = data['rating'].mean()
            std_dev_rating = data['rating'].std()

            # Print the results
            print(f"Mean of 'rating': {mean_rating:.2f}")
            print(f"Standard Deviation of 'rating': {std_dev_rating:.2f}")
            print('-' * 50)
        else:
            print("The column 'rating' is not present in the dataset.")
    except Exception as e:
        print(f"An error occurred while processing {file_name}: {e}")

print("Processing complete.")