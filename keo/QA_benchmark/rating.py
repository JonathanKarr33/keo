import pandas as pd

# Load the CSV file
file_name_list = ["GraphRAG_QA_Evaluation.csv", "GraphRAG_spacy_QA_Evaluation.csv", "VanillaGPT4o_QA_Evaluation.csv"]
for file_name in file_name_list:
    data = pd.read_csv(file_name)

    # Print the file name without the extension
    print(file_name.split(".")[0])

    # Calculate the mean and standard deviation of the 'rating' column
    if 'rating' in data.columns:
        mean_rating = data['rating'].mean()
        std_dev_rating = data['rating'].std()

        # Print the results
        print(f"Mean of 'rating': {mean_rating:.2f}")
        print(f"Standard Deviation of 'rating': {std_dev_rating:.2f}")
    else:
        print("The column 'rating' is not present in the dataset.")
