import pandas as pd
import re

# 正则表达式匹配标准的电话号码格式（美国）
def is_valid_phone_number(phone):
    return re.match(r"^\d{3}-\d{3}-\d{4}$", phone) is not None

# 正则表达式匹配标准的电子邮件格式
def is_valid_email(email):
    return re.match(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", email) is not None

def is_valid_linkedin(linkedin):
    return "linkedin.com" in linkedin

def clean_job(job_name):
    return job_name.replace("Generate a Resume for a ", "")

# 尝试使用不同的编码格式读取数据
def read_data(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_path, encoding='ISO-8859-1')
        except UnicodeDecodeError:
            return pd.read_csv(file_path, encoding='cp1252')

file_path = r'D:/Final_Project_Resume/Resume_dataset/Name_dataset.csv'
data = read_data(file_path)

# 处理 NA / NaN 值
data = data.dropna(subset=['JOB NAME', 'Resume', 'First Name', 'Last Name', 'Phone Number', 'Email', 'LinkedIn Website'])

# 过滤坏数据
data = data[data['JOB NAME'].str.startswith('Generate')]
data = data[data['Resume'].str.len() >= 30]
data = data[data['First Name'].apply(lambda x: isinstance(x, str))]
data = data[data['Last Name'].apply(lambda x: isinstance(x, str))]
data = data[data['Phone Number'].apply(is_valid_phone_number)]
data = data[data['Email'].apply(is_valid_email)]
data = data[data['LinkedIn Website'].apply(is_valid_linkedin)]

# 数据格式化
data['Phone Number'] = data['Phone Number'].apply(lambda x: re.sub(r"(\d{3})-(\d{3})-(\d{4})", r"(\1) \2-\3", x))
data['JOB NAME'] = data['JOB NAME'].apply(clean_job)

# 保存清理后的数据
output_path = r"D:/Final_Project_Resume/Resume_dataset/Clear_Name_dataset.csv"
data.to_csv(output_path, index=False)
