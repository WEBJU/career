from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model and encoder
model = joblib.load('career_model.pkl')
encoder = joblib.load('encoder.pkl')
columns = joblib.load('career_colums.pkl')

# Define category mappings
CRM_Managerial_Roles = ['CRM Business Analyst', 'CRM Technical Developer', 'Project Manager', 'Information Technology Manager']
Analyst = ['Business Systems Analyst', 'Business Intelligence Analyst', 'E-Commerce Analyst']
Mobile_Applications_Web_Development = ['Mobile Applications Developer', 'Web Developer', 'Applications Developer']
QA_Testing = ['Software Quality Assurance (QA) / Testing', 'Quality Assurance Associate']
UX_Design = ['UX Designer', 'Design & UX']
Databases = ['Database Developer', 'Database Administrator', 'Database Manager', 'Portal Administrator']
Programming_Systems_Analyst = ['Programmer Analyst', 'Systems Analyst']
Networks_Systems = ['Network Security Administrator', 'Network Security Engineer', 'Network Engineer',
                    'Systems Security Administrator', 'Software Systems Engineer', 'Information Security Analyst']
SE_SDE = ['Software Engineer', 'Software Developer']
Technical_Support_Service = ['Technical Engineer', 'Technical Services/Help Desk/Tech Support', 'Technical Support']
others = ['Solutions Architect', 'Data Architect', 'Information Technology Auditor']

# Map categories to their respective roles
category_mappings = {
    'CRM/Managerial Roles': CRM_Managerial_Roles,
    'Analyst': Analyst,
    'Mobile Applications/ Web Development': Mobile_Applications_Web_Development,
    'QA/Testing': QA_Testing,
    'UX/Design': UX_Design,
    'Databases': Databases,
    'Programming/ Systems Analyst': Programming_Systems_Analyst,
    'Networks/ Systems': Networks_Systems,
    'SE/SDE': SE_SDE,
    'Technical Support/Service': Technical_Support_Service,
    'others': others
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        data = request.json.get('data')
        if data is None:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert the data to a DataFrame
        df = pd.DataFrame([data], columns=columns)
        
        # Perform one-hot encoding
        encoded_data = encoder.transform(df)
        
        # Make predictions
        predictions = model.predict(encoded_data)
        
        # Get the category and roles
        category = predictions[0]
        roles = category_mappings.get(category, [])
        
        return jsonify({'category': category, 'roles': roles})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
