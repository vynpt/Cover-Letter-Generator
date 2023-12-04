#import nltk
import math
import pandas as pd

pf_name = 'D:/Final_Project_Resume/Resume_dataset/Clear_Name_dataset.csv'

df = pd.read_csv(pf_name)

print(list(df['Resume'])[:10000])


