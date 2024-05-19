from flask import Flask, request, jsonify
import tensorflow_hub as hub
import tensorflow_text  # Required for using Universal Sentence Encoder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def preprocess_text(text):
    # Preprocessing steps (e.g., lowercasing, punctuation removal) can be added here if needed
    return text


def calculate_similarity(teacher_answer, student_answer):
    # Preprocess text
    teacher_answer = preprocess_text(teacher_answer)
    student_answer = preprocess_text(student_answer)

    # Encode sentences
    teacher_embedding = embed([teacher_answer])[0]
    student_embedding = embed([student_answer])[0]

    # Calculate cosine similarity
    similarity_score = cosine_similarity(
        [teacher_embedding], [student_embedding])[0][0]
    return similarity_score


@app.route('/check_similarity', methods=['POST'])
def check_similarity():
    data = request.get_json()
    teacher_answer = data['teacher_answer']
    student_answer = data['student_answer']

    # Calculate similarity
    similarity_score = calculate_similarity(teacher_answer, student_answer)

    # You may adjust the threshold as needed
    threshold = 0.7

    # Determine similarity result
    if similarity_score >= threshold:
        similarity_result = "Similar"
    else:
        similarity_result = "Not Similar"

    response = {
        "similarity_score": similarity_score,
        "similarity_result": similarity_result
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
