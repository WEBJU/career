from flask import Flask, request, jsonify
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse
import pandas as pd


app = Flask(__name__)

# Load the trained model
clf = pickle.load(open("final_model.sav", 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json["data"]

    # Convert numerical values to integers and categorical values to strings
    data = [str(val) if isinstance(val, str) else int(val) for val in data]

    # Transform the input data
    x = pd.DataFrame([data], columns=range(len(data)))  # Use numerical indices
    encoder = pickle.load(open("encoder.pkl", 'rb'))
    new_data = encoder.fit_transform(x)
    new_data = StandardScaler(with_mean=False).fit_transform(new_data)
    new_data = sparse.csr_matrix.copy(new_data)

    # Make prediction
    output = clf.predict(new_data)
    pred_prob = clf.predict_proba(new_data)

    # Convert output to string
    suggested_role = output[0]

    # Prepare response
    response = {
        'predicted_role': suggested_role,
        'confidence_scores': {role: prob for role, prob in zip(clf.classes_, pred_prob[0])}
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
