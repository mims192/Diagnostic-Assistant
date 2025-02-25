import streamlit as st
import joblib
import numpy as np
import google.generativeai as genai
from api_key import api_key
import speech_recognition as sr


# Configure API Key
genai.configure(api_key=api_key)

# Generation Configuration
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Safety Settings
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
]

# System Prompt
system_prompts = """As a highly skilled medical practitioner specializing in image analysis, you are tasked with examining medical images for a renowned hospital. Your expertise is crucial in identifying any anomalies, diseases, or health issues that may be present in the images.

Your Responsibilities include:
1. **Detailed Analysis**: Thoroughly analyze each image, focusing on identifying any abnormal findings.
2. **Findings Report**: Document all observed anomalies or signs of disease. Clearly articulate these findings in a structured format.
3. **Recommendations and Next Steps**: Based on your analysis, suggest potential next steps, including further tests or treatments as applicable.
4. **Treatment Suggestions**: If appropriate, recommend possible treatment options or interventions.

**Important Notes**:
- Only respond if the image pertains to human health issues.
- If image quality is unclear, note that certain aspects are "Unable to be determined based on the provided image."
- Include a disclaimer: "Consult with a Doctor before making any decisions."
"""
Prompts="""
As a highly skilled medical expert specializing in text-based medical analysis, your role involves assessing detailed descriptions of medical conditions, symptoms, and diagnostic findings. Your expertise is essential in identifying potential health concerns and providing structured insights.

Your Responsibilities:
1.Thorough Evaluation: Carefully analyze the provided medical information, focusing on identifying any abnormalities, diseases, or concerns.
2.Findings Report: Document all observations in a clear and structured format, highlighting any potential health risks.
3.Recommendations & Next Steps: Based on your assessment, suggest appropriate follow-up actions, such as additional tests or specialist consultations.
4.Treatment Suggestions: If applicable, recommend possible treatment options or interventions based on the information provided.
Important Notes:
1.Only respond if the text pertains to human health-related concerns.
2.If the provided information is insufficient for an accurate assessment, indicate that certain aspects are "unable to be determined based on the available details."
3.Disclaimer: Always advise consulting a medical professional before making any health-related decisions.
"""


# Load ML Model & Scaler
model_path = "random_forest_model.pkl"
scaler_path = "scaler.pkl"

classifier = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Initialize Model
genai_model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Streamlit UI
st.set_page_config(page_title="Medical Diagnostic Assistant", page_icon="🩺")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Image Analysis", "Blood Report Analysis"])

# Main Page
if page == "Dashboard":
    st.title("🩺 Medical Diagnostic Assistant")
    st.write("Welcome to the **Medical Diagnostic Assistant**. This system helps healthcare professionals analyze medical images, patient data, and symptoms for **accurate diagnosis**.")
    
    model = genai.GenerativeModel("gemini-pro")

    st.title("How are you feeling today?")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Function to capture voice input
    def recognize_speech():
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.write("🎤 Listening... Speak now!")
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return "Sorry, I couldn't understand that."
            except sr.RequestError:
                return "Speech Recognition service is unavailable."

    # Button for voice input
    if st.button("🎙️ Speak"):
        user_voice_input = recognize_speech()
        if user_voice_input:
            st.session_state.messages.append({"role": "user", "content": user_voice_input})
            st.write(f"🗣️ You: {user_voice_input}")

            # Get AI response based on  prompt
            full_prompt = Prompts + "\nUser: " + user_voice_input
            response = model.generate_content(full_prompt)
            ai_reply = response.text

            # Add AI response to chat history
            st.session_state.messages.append({"role": "assistant", "content": ai_reply})

            # Display AI response
            with st.chat_message("assistant"):
                st.write(ai_reply)

    # Text input field
    user_text_input = st.chat_input("Type your message here...")

    if user_text_input:
        st.session_state.messages.append({"role": "user", "content": user_text_input})

        # Get AI response based on prompt
        full_prompt = Prompts + "\nUser: " + user_text_input
        response = model.generate_content(full_prompt)
        ai_reply = response.text

        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_reply})

        # Display AI response
        with st.chat_message("assistant"):
            st.write(ai_reply)

# Image Analysis Page
elif page == "Image Analysis":
    st.title("Medical Image Analysis")
    st.write("Upload a medical image for analysis.")

    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if st.button("Generate Analysis"):
        if uploaded_file:
            mime_type = f"image/{uploaded_file.type.split('/')[-1]}"
            image_data = uploaded_file.getvalue()
            image_parts = [{"mime_type": mime_type, "data": image_data}]
            prompt_parts = [image_parts[0], system_prompts]

            response = genai_model.generate_content(prompt_parts)
            st.write(response.text)
        else:
            st.warning("Please upload an image before generating analysis.")

# Diagnostics Page (Blood Report Analysis)
elif page == "Blood Report Analysis":
    st.title("🔬 Blood Report Analysis")
    st.subheader("Enter Patient Details")

    gender = st.selectbox("Gender", ["Male", "Female", "Child"])
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    haemoglobin = st.number_input("Haemoglobin Level", min_value=0.0, max_value=20.0, step=0.1)
    platelets = st.number_input("Platelet Count", min_value=0, max_value=800000, step=1000)
    wbc = st.number_input("WBC Count", min_value=0, max_value=50000, step=500)

    # Gender Encoding
    gender_dict = {"child": [1.0, 0.0, 0.0], "female": [0.0, 1.0, 0.0], "male": [0.0, 0.0, 1.0]}
    gender_encoded = gender_dict[gender.lower()]

    if st.button("Predict Blood Condition"):
        input_data = np.array([gender_encoded + [age, haemoglobin, platelets, wbc]]).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)
        prediction = classifier.predict(input_data_scaled)

        disease_mapping = {0: "Healthy", 1: "Anemia", 2: "Leukemia", 3: "Thrombocytopenia"}
        result = disease_mapping.get(prediction[0], "Unknown Condition")

        st.success(f"Prediction: {result}")
