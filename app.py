from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)

# Load the data
data = pd.read_csv('dataset_final.csv')

# Initialize BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings for a given text
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Concatenate all textual features into one string for BERT embeddings
textual_features = ['Carrer_Goal', 'Location', 'Hobby', 'Language', 'Skills', 'Lifestyle', 'Celebrity', 'Education', 'College']
data['combined_text'] = data[textual_features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# Get BERT embeddings for all users
embeddings = np.vstack(data['combined_text'].apply(get_bert_embedding))

# Encode categorical variables for numerical processing
label_encoders = {}
for column in ['Skills', 'Carrer_Goal', 'Language', 'College', 'Hobby', 'Gender', 'Education', 'Location', 'Lifestyle', 'Celebrity']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Define weights
weights = {
    'Age': 0.3,
    'Skills': 0.2,
    'Carrer_Goal': 0.15,
    'Language': 0.1,
    'College': 0.05,
    'Hobby': 0.05,
    'Gender': 0.05,
    'Education': 0.03,
    'Location': 0.03,
    'Lifestyle': 0.02,
    'Celebrity': 0.02,
}

# Select and scale the numerical data
numerical_features = ['Age', 'Skills', 'Carrer_Goal', 'Language', 'College', 'Hobby', 'Gender', 'Education', 'Location', 'Lifestyle', 'Celebrity']
numerical_data = data[numerical_features]
scaler = StandardScaler()
numerical_data_scaled = scaler.fit_transform(numerical_data)

# Apply weights to the numerical data
weighted_numerical_data = numerical_data_scaled * np.array(list(weights.values()))

# Combine numerical data and BERT embeddings
combined_data = np.hstack((weighted_numerical_data, embeddings))

# Calculatee Cosine similarity
similarity_matrix = cosine_similarity(combined_data)

# Function to find buddies for a given user_id
def find_buddies(user_id, data, similarity_matrix, top_n=3):
    user_index = data[data['user_id'] == user_id].index[0]
    similarity_scores = list(enumerate(similarity_matrix[user_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_users = [data.iloc[i[0]]['user_id'].item() for i in similarity_scores[1:top_n+1]] 
    return similar_users

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/find_buddies', methods=['GET'])
def find_buddies_route():
    user_id = int(request.args.get('user_id'))
    top_n = int(request.args.get('top_n', 3))
    buddies = find_buddies(user_id, data, similarity_matrix, top_n)
    return jsonify({'user_id': user_id, 'buddies': buddies})

if __name__ == '__main__':
    app.run(debug=True)
