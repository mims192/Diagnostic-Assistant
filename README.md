Medical Diagnostic Assistant

Overview-:

The Medical Diagnostic Assistant is an AI-powered healthcare application built using Streamlit. It leverages Google's Gemini AI, Machine Learning models, and Speech Recognition to analyze medical images, patient symptoms, and blood reports to provide diagnostic insights.

Features-:

1. Dashboard: AI-powered chatbot for symptom-based medical consultation.

2. Image Analysis: Upload and analyze medical images using AI.

3. Blood Report Analysis: Predicts health conditions based on blood test results.

4. Speech Recognition: Allows users to interact with the chatbot via voice input.

Technologies Used-:

1. Python

2. Streamlit (for UI development)

3. Google Generative AI (Gemini)

4. Joblib (for model loading)

5. NumPy (for data processing)

6. SpeechRecognition (for voice input handling)

Installation-:

Prerequisites-:

1. Ensure you have Python installed (>=3.8).

Setup Instructions-:

1. Clone the repository:

git clone <repository_url>
cd <project_directory>

2. Install dependencies:

pip install -r requirements.txt

3. Add your Google Gemini API Key in api_key.py:

api_key = "your_google_api_key_here"

4. Run the application:

streamlit run app.py

Usage-:

1.Dashboard (Chatbot)

Users can type or speak their symptoms.

The chatbot provides analysis and recommendations based on medical AI.

2.Image Analysis

Upload a medical image (PNG, JPG, JPEG).

AI processes the image and provides a detailed report.

3.Blood Report Analysis

Enter gender, age, haemoglobin, platelet count, WBC count.

ML model predicts health condition (e.g., Anemia, Leukemia).

Model Information-:

1.Uses a Random Forest Model for blood report analysis.

2.Utilizes Google's Gemini AI for text and image-based diagnosis.

Future Enhancements:
1. Personalized Health Reports: Provide detailed patient-specific health insights and progress tracking.
2. Integration with Wearable Devices: Collect real-time health data from smartwatches and fitness trackers.
3. Multi-Language Support: Enable chatbot interactions in multiple languages for broader accessibility.
4. Suggest potential medications and treatments based on AI-driven diagnosis.
5. Implement an AI-driven therapy assistant to provide mental health support, guided meditation, stress management techniques, and mood tracking.
6. Cloud Storage for Reports i.e securely store and retrieve medical reports for future reference.
