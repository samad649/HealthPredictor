from flask import Flask, request, jsonify
from predictor import NaiveBayes  
from flask_cors import CORS
app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)
# Initialize the Naive Bayes model
nb_model = NaiveBayes()
print("made model")
nb_model.process_data()
print("processed data")
nb_model.train()
print("model has been trained")

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json.get('input_text')

    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    # Use the model to predict the diagnosis
    predictions = nb_model.predict([input_text])  # Predict for the given text

    # Print the predictions to see what they look like
    print(f"Predictions: {predictions}")

    # Check if predictions is a tuple and extract the first element
    if isinstance(predictions, tuple):
        predictions = predictions[0]  # Extract the first element if it's a tuple

    # If predictions is a numpy array, you can use tolist()
    print(f"Predictions after processing: {predictions}")

    # Return the predictions as a response
    return jsonify({"diagnoses": predictions})
    

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)