from flask import Flask, request, jsonify
import joblib
import random
import pandas as pd
from flask import Flask
from flask_compress import Compress

app = Flask(__name__)

# Load the saved model, vectorizer, and label encoder
model = joblib.load('chatbot_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Load questions and options from CSV file
df = pd.read_csv('phenotype.csv')
questions = df['question'].tolist()
options = df.groupby('question')['option'].apply(list).to_dict()

@app.route('/ask_question', methods=['GET'])
def ask_question():
    # Generate a random question and options
    index = random.randint(0, len(questions) - 1)
    question = questions[index]
    option_list = options[question]

    # Return the question and options as JSON
    return jsonify({"question": question, "options": option_list})

@app.route('/process_response', methods=['POST'])
def process_response():
    # Receive user response and process it
    data = request.get_json()
    user_response = data.get("response")

    # Perform classification using the loaded model
    processed_user_response = vectorizer.transform([user_response])
    predicted_class = model.predict(processed_user_response)[0]
    class_label = label_encoder.inverse_transform([predicted_class])[0]

    # Return the classification result as JSON
    result = {"constitution": class_label}
    return jsonify(result)

Compress(app)

if __name__ == '__main__':
    app.run(debug=True)
