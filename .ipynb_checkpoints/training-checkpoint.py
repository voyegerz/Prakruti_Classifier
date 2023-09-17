import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# step 2
# Load the dataset
df = pd.read_csv('new.csv')

# Preprocess text data
df['question'] = df['question'].str.lower()
df['option'] = df['option'].str.lower()

# Encode the 'class' column
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])

# Tokenize the 'question' and 'option' columns
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['question'] + df['option'])
df['question_sequence'] = tokenizer.texts_to_sequences(df['question'])
df['option_sequence'] = tokenizer.texts_to_sequences(df['option'])

# Pad sequences
max_sequence_length = max(len(seq) for seq in df['question_sequence'])
df['question_sequence'] = pad_sequences(df['question_sequence'], maxlen=max_sequence_length)
df['option_sequence'] = pad_sequences(df['option_sequence'], maxlen=max_sequence_length)

# step 3
# Combine question and option sequences
X = np.hstack((df['question_sequence'].values, df['option_sequence'].values))

# Reshape X to match the expected input shape
X = X.reshape(X.shape[0], -1)

# Encode the 'class' column
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])
y = to_categorical(df['class'], num_classes=len(label_encoder.classes_))

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


# step 4
# Define the RNN model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_sequence_length*2))
model.add(LSTM(128))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# step 5
# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
