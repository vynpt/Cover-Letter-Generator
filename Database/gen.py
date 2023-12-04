import pandas as pd
from faker import Faker

# Initialize Faker
fake = Faker()

# Function to generate fake data
def generate_fake_data():
    first_name = fake.first_name()
    last_name = fake.last_name()
    phone_number = fake.numerify(text="###-###-####")  # Generate phone number in XXX-XXX-XXXX format
    email_domain = fake.random_element(elements=['gmail.com', 'outlook.com', 'hotmail.com'])
    email = fake.user_name() + '@' + email_domain  # Generate a random email
    linkedin_website = f"https://www.linkedin.com/in/{first_name}-{last_name}".lower()  # Generate LinkedIn URL

    return first_name, last_name, phone_number, email, linkedin_website

# Read CSV file
file_path = 'C:/Users/Texas2020/Desktop/NLP/resume/updated_data_final_cleaned.csv'
output_path = 'C:/Users/Texas2020/Desktop/NLP/resume/dataset.csv'
try:
    df = pd.read_csv(file_path)

    # Generate fake data for each row in the dataframe
    fake_data = [generate_fake_data() for _ in range(len(df))]

    # Convert the list of tuples to a dataframe
    fake_data_df = pd.DataFrame(fake_data, columns=['First Name', 'Last Name', 'Phone Number', 'Email', 'LinkedIn Website'])

    # Combine the original and the fake data dataframes
    combined_df = pd.concat([df, fake_data_df], axis=1)

    # Save the combined dataframe to the specified path
    combined_df.to_csv(output_path, index=False)

    print(f"Fake data added and saved to '{output_path}'.")
except FileNotFoundError:
    print("File not found. Please check the file path.")
except Exception as e:
    print(f"An error occurred: {e}")
