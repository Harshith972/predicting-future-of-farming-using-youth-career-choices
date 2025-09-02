import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from flask import Flask, request, jsonify, render_template_string
import warnings

# Suppress all future warnings for a cleaner console
warnings.filterwarnings('ignore', category=FutureWarning)

app = Flask(__name__)

# --- Global Variables & Configuration ---

# This will store the trained model pipeline once it's loaded.
model_pipeline = None

# A dictionary to map the new CSV column names to the standardized ones used by the model.
COLUMN_NAME_MAP = {
    'Age': 'age',
    'Gender': 'gender',
    'Education Level': 'education_level',
    # Note: The 'State' column from the new CSV is not used as a feature.
    'Owns Agricultural Land': 'has_land',
    'Land in Acres': 'land_acres',
    "Parents' Occupation": 'parents_occupation',
    'Career Interest': 'interested_career',
    'Interested in Farming': 'is_interested_in_farming'
}

# Defines the exact features and their order that the model expects.
# 'location' (Rural/Urban) has been removed as it's not in the new CSV.
FEATURE_COLUMNS = [
    'age', 'gender', 'education_level', 'has_land', 
    'land_acres', 'parents_occupation', 'interested_career'
]
    
# Lists the categorical features that need to be transformed using one-hot encoding.
# 'location' has been removed.
CATEGORICAL_FEATURES = [
    'gender', 'education_level', 'has_land', 
    'parents_occupation', 'interested_career'
]

# --- HTML & JavaScript Frontend ---

# The HTML template has been updated to remove the "Location" dropdown.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Future of Farming Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f3f4f6;
        }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen p-4">

    <!-- Main Container -->
    <div class="w-full max-w-2xl bg-white rounded-xl shadow-2xl p-8 space-y-8 border border-gray-200">
        <!-- Header -->
        <header class="text-center space-y-2">
            <h1 class="text-4xl font-extrabold text-gray-800">Future of Farming Predictor</h1>
            <p class="text-lg text-gray-500">Predicting youth's likelihood of choosing a farming career.</p>
        </header>

        <!-- Prediction Form -->
        <section class="space-y-6">
            <h2 class="text-2xl font-semibold text-gray-700">Enter a hypothetical profile:</h2>
            <form id="prediction-form" class="grid grid-cols-1 md:grid-cols-2 gap-6">

                <!-- Age -->
                <div class="relative">
                    <label for="age" class="block text-sm font-medium text-gray-700 mb-1">Age</label>
                    <input type="number" name="age" id="age" value="22" min="15" max="40" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-colors duration-200">
                </div>
                
                <!-- Gender -->
                <div class="relative">
                    <label for="gender" class="block text-sm font-medium text-gray-700 mb-1">Gender</label>
                    <select name="gender" id="gender" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-colors duration-200">
                        <option>Male</option>
                        <option>Female</option>
                        <option>Other</option>
                    </select>
                </div>

                <!-- Education Level -->
                <div class="relative">
                    <label for="education_level" class="block text-sm font-medium text-gray-700 mb-1">Education Level</label>
                    <select name="education_level" id="education_level" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-colors duration-200">
                        <option>School</option>
                        <option>Intermediate</option>
                        <option>Undergraduate</option>
                        <option>Postgraduate</option>
                        <option>PhD</option>
                    </select>
                </div>
                
                <!-- Parents Occupation -->
                <div class="relative">
                    <label for="parents_occupation" class="block text-sm font-medium text-gray-700 mb-1">Parents' Occupation</label>
                    <select name="parents_occupation" id="parents_occupation" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-colors duration-200">
                        <option>Farming</option>
                        <option>Daily Wage Labour</option>
                        <option>Government Job</option>
                        <option>Private Job</option>
                        <option>Business</option>
                    </select>
                </div>

                <!-- Interested Career -->
                <div class="relative">
                    <label for="interested_career" class="block text-sm font-medium text-gray-700 mb-1">Interested Career</label>
                    <select name="interested_career" id="interested_career" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-colors duration-200">
                        <option>Farmer</option>
                        <option>Software Engineer</option>
                        <option>Teacher</option>
                        <option>Doctor</option>
                        <option>Government Job</option>
                        <option>Scientist</option>
                        <option>Entrepreneur</option>
                    </select>
                </div>

                <!-- Access to Land -->
                <div class="relative">
                    <label for="has_land" class="block text-sm font-medium text-gray-700 mb-1">Access to Land?</label>
                    <select name="has_land" id="has_land" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-colors duration-200">
                        <option>Yes</option>
                        <option>No</option>
                    </select>
                </div>

                <!-- Land Acres -->
                <div class="relative md:col-span-2">
                    <label for="land_acres" class="block text-sm font-medium text-gray-700 mb-1">Land Owned (acres)</label>
                    <input type="number" name="land_acres" id="land_acres" value="5" min="0" step="0.1" class="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500 transition-colors duration-200">
                </div>

            </form>

            <!-- Predict Button -->
            <div class="text-center">
                <button onclick="predictCareerChoice()" class="w-full md:w-auto px-6 py-3 bg-green-600 text-white font-bold rounded-lg shadow-lg hover:bg-green-700 focus:outline-none focus:ring-4 focus:ring-green-500 focus:ring-opacity-50 transition-transform duration-200 transform hover:scale-105">
                    Predict Career Choice
                </button>
            </div>
        </section>
        
        <!-- Prediction Output -->
        <section id="result-container" class="mt-8 hidden">
            <h2 class="text-2xl font-semibold text-gray-700 text-center mb-4">Prediction Results</h2>
            <div class="p-6 rounded-lg shadow-lg text-center border-2 border-green-500 bg-green-50">
                <p id="prediction-result" class="text-2xl font-bold text-gray-800 mb-2"></p>
                <p id="prediction-confidence" class="text-lg text-gray-600"></p>
            </div>
        </section>

    </div>

    <!-- JavaScript to handle the logic -->
    <script>
        async function predictCareerChoice() {
            // Collect data from the form.
            const profile = {
                age: parseInt(document.getElementById('age').value),
                gender: document.getElementById('gender').value,
                education_level: document.getElementById('education_level').value,
                parents_occupation: document.getElementById('parents_occupation').value,
                interested_career: document.getElementById('interested_career').value,
                has_land: document.getElementById('has_land').value,
                land_acres: parseFloat(document.getElementById('land_acres').value),
            };
            
            if (isNaN(profile.age) || isNaN(profile.land_acres)) {
                alert("Please enter valid numbers for Age and Land Owned.");
                return;
            }

            // Show a loading state
            document.getElementById('prediction-result').textContent = 'Predicting...';
            document.getElementById('prediction-confidence').textContent = '';
            document.getElementById('result-container').classList.remove('hidden');

            try {
                // Send data to the Flask backend
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(profile)
                });
                
                if (!response.ok) throw new Error(`Server error: ${response.status}`);

                const data = await response.json();

                // Display the results
                document.getElementById('prediction-result').textContent = `Prediction: ${data.prediction_text}`;
                document.getElementById('prediction-confidence').textContent = `Confidence: ${Math.round(data.confidence * 100)}%`;

            } catch (error) {
                alert('Failed to get prediction. Check the server console for errors.');
                console.error('Error:', error);
            }
        }
    </script>
</body>
</html>
"""

# --- Model Training Function ---

def train_model(filepath):
    """
    Loads data, preprocesses it, and trains the model.
    """
    print(f"\nLoading and training model from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    
    # --- Data Cleaning and Preprocessing ---
    df.rename(columns=COLUMN_NAME_MAP, inplace=True)
    
    # Handle potential missing values in land_acres by filling with 0
    df['land_acres'] = df['land_acres'].fillna(0)
    
    # Convert target variable to binary (1 for 'Yes', 0 for anything else)
    df['is_interested_in_farming'] = df['is_interested_in_farming'].apply(
        lambda x: 1 if str(x).strip() == 'Yes' else 0
    )
    
    # Ensure all feature columns exist, fill missing with a placeholder if necessary
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 'Unknown' # Or some other default

    # Separate features (X) and the target variable (y)
    X = df[FEATURE_COLUMNS]
    y = df['is_interested_in_farming']

    # --- Model Pipeline Creation ---
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    print("Model training initiated...")
    pipeline.fit(X, y)
    print("Model training complete. Server is ready.")
    return pipeline

# --- Flask Application Setup & Routes ---

@app.before_request
def setup():
    """
    Trains the model once before the first request.
    """
    global model_pipeline
    if model_pipeline is None:
        model_pipeline = train_model('response_500.csv')

@app.route('/')
def home():
    """Serves the main HTML page."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the frontend."""
    if model_pipeline is None:
        return jsonify({'error': 'Model is not trained yet.'}), 503

    try:
        user_data = request.json
        input_df = pd.DataFrame([user_data])
        
        # ** FIX: Ensure the column order matches the training order. **
        input_df = input_df[FEATURE_COLUMNS]
        
        # Use the pipeline to make a prediction
        prediction_proba = model_pipeline.predict_proba(input_df)[0]
        prediction_class = int(model_pipeline.predict(input_df)[0]) # Ensure it's a standard int
        
        # Format the response
        prediction_text = "Choosing a Farming Career" if prediction_class == 1 else "Not Choosing a Farming Career"
        confidence = prediction_proba[prediction_class]

        return jsonify({
            'prediction_text': prediction_text,
            'confidence': float(confidence)
        })

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# --- Main Execution Block ---

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

