import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib


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

# Save the trained model to a file
joblib.dump(model, 'chatbot_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
