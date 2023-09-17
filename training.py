import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import random

# Load the dataset
df = pd.read_csv('phenotype.csv')

# Encode the 'class' column
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

# Tokenize the 'question' and 'option' columns
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['question'] + " " + df['option'])

# Train a model (e.g., Logistic Regression)
model = LogisticRegression()
model.fit(X, df['class'])

# Define a function to ask a random set of questions with options
def ask_random_questions_with_options(num_questions=20):
    random_questions = random.sample(df['question'].tolist(), num_questions)
    user_responses = []
    
    for i, question_text in enumerate(random_questions, 1):
        question_options = df[df['question'] == question_text]
        
        # Select the top three options based on their relevance to the class
        question_options = question_options.nlargest(3, 'class')
        
        options = question_options['option'].tolist()
        class_labels = question_options['class'].tolist()
        
        print(f"Question {i}: {question_text}")
        for j, option in enumerate(options, 1):
            print(f"{j}. {option}")
        
        user_input = input("Select an option (1, 2, or 3): ")
        
        try:
            selected_option = options[int(user_input) - 1]
            selected_class = class_labels[int(user_input) - 1]
            user_responses.append((selected_option, selected_class))
        except (ValueError, IndexError):
            print("Invalid option. Please select 1, 2, or 3.")
    
    return user_responses

# Ask random questions with options
print("Please answer the following questions (select options 1, 2, or 3):")
user_responses = ask_random_questions_with_options()

# Process user responses
processed_user_responses = [response[0] for response in user_responses]

# Classify user based on responses
processed_user_responses = vectorizer.transform(processed_user_responses)
predicted_class = model.predict(processed_user_responses)

# Calculate class percentages
class_counts = pd.Series(predicted_class).value_counts()
total_responses = len(predicted_class)
class_percentages = class_counts / total_responses * 100

# Map class labels back to their original labels
class_labels = label_encoder.inverse_transform(class_counts.index)

# Display class percentages
for label, percentage in zip(class_labels, class_percentages):
    print(f"{label}: {percentage:.2f}%")
