import pandas as pd
import streamlit as st

jobs = pd.read_csv("Database/Job_targets.csv", encoding='ISO-8859-1')
columns_to_remove = [0, 1, 5, 6, 10, 12, 13, 17, 18, 21, 23, 27, 28, 29, 30, 31, 33, 34, 36, 37, 38, 39, 40]

# Drop the empty or unnecessary columns 
column_labels_to_remove = jobs.columns[columns_to_remove]
jobs = jobs.drop(column_labels_to_remove, axis=1)

# Drop the nan value in job title
jobs = jobs.dropna(subset=['title'])

print(jobs['title'].unique())

jobs.to_csv('/Users/vynguyen/Desktop/CS505/Cover-Letter-Generator/Database/Job_cleaned.csv', index=False)



