import json
import re
import os
import numpy as np
import joblib
import speech_recognition as sr
from sklearn.ensemble import RandomForestClassifier
import easyocr
import tempfile
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

class PrescriptionManager:
    def __init__(self, file_path='prescriptions.json'):
        self.file_path = file_path
        self.prescriptions = self.load_prescriptions()

    def load_prescriptions(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as file:
                data = json.load(file)
            return data
        else:
            return {}

    def get_patient_prescription(self, patient_id):
        return self.prescriptions.get(patient_id, None)

    def update_prescription(self, patient_id, new_data):
        if patient_id in self.prescriptions:
            self.prescriptions[patient_id].update(new_data)
        else:
            self.prescriptions[patient_id] = new_data
        self.save_prescriptions()

    def save_prescriptions(self):
        with open(self.file_path, 'w') as file:
            json.dump(self.prescriptions, file, indent=4)

class InputProcessor:
    def __init__(self):
        # We keep the recognizer here, but note that for live voice input we use client-side JS.
        self.recognizer = sr.Recognizer()

    def extract_parameters(self, text):
        parameters = {}
        pattern_dict = {
            'diabetes': [r"(?:diabetes|blood sugar)\s*(?::|-)?\s*(\d+|low|normal|high)"],
            'blood_pressure': [r"blood pressure\s*(?::|-)?\s*([0-9]{2,3}(?:/[0-9]{2,3})?)"],
            'age': [r"age\s*(?::|-)?\s*(\d+)", r"(\d+)\s*(?:years old|yrs?)"],
            'weight': [r"(?:weight|wt)\s*(?::|-)?\s*(\d+\s*(?:kg|kgs|kilograms)?)", r"(\d+\s*(?:kg|kgs|kilograms))"],
            'spo2': [r"(?:SPO2|SOP)\s*(?::|-)?\s*(\d{2,3})%?", r"(\d{2,3})\s*%"]
        }
        for key, patterns in pattern_dict.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    parameters[key] = match.group(1)
                    break
        return parameters

    def perform_ocr(self, image_path):
        try:
            reader = easyocr.Reader(['en'], gpu=False)
            ocr_results = reader.readtext(image_path, detail=1)
            ocr_results = sorted(ocr_results, key=lambda x: (x[0][0][1], x[0][0][0]))
            ocr_text = "\n".join([result[1] for result in ocr_results])
            ocr_text = re.sub(r'\n+', '\n', ocr_text).strip()
            return ocr_text
        except Exception as e:
            print("Error processing the image with EasyOCR:", e)
            return ""

class MLPredictor:
    def __init__(self, model_path='ml_model.pkl'):
        self.model_path = model_path
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
        else:
            self.model = self.train_dummy_model()
            joblib.dump(self.model, model_path)

    def train_dummy_model(self):
        X = []
        y = []
        levels = {'low': 0, 'normal': 1, 'high': 2}
        for _ in range(100):
            d_level = np.random.choice(list(levels.values()))
            systolic = np.random.randint(100, 180)
            spo2 = np.random.randint(90, 100)
            risk = 1 if (systolic > 140 or spo2 < 95 or d_level == 2) else 0
            X.append([d_level, systolic, spo2])
            y.append(risk)
        clf = RandomForestClassifier()
        clf.fit(X, y)
        return clf

    def convert_blood_sugar_to_category(self, blood_sugar):
        try:
            bs = int(blood_sugar)
            if bs < 80:
                return 'low'
            elif bs <= 140:
                return 'normal'
            else:
                return 'high'
        except:
            return blood_sugar.lower()

    def predict_disease_risk(self, parameters):
        levels = {'low': 0, 'normal': 1, 'high': 2}
        d_value = parameters.get('diabetes', 'normal')
        if d_value.isdigit():
            d_value = self.convert_blood_sugar_to_category(d_value)
        else:
            d_value = d_value.lower()
        bp = parameters.get('blood_pressure', '120/80')
        if not bp.strip():
            bp = '120/80'
        try:
            systolic = int(bp.split('/')[0]) if '/' in bp else int(bp)
        except ValueError:
            systolic = 120
        spo2_str = parameters.get('spo2', '98')
        if not spo2_str.strip():
            spo2_str = '98'
        try:
            spo2 = int(spo2_str)
        except ValueError:
            spo2 = 98
        features = np.array([[levels.get(d_value, 1), systolic, spo2]])
        risk = int(self.model.predict(features)[0])
        print(f"Predicted disease risk (0 = low risk, 1 = high risk): {risk}")
        return risk

class EHRManager:
    def __init__(self, ehr_file='ehr_records.json'):
        self.ehr_file = ehr_file
        self.records = self.load_records()

    def load_records(self):
        if os.path.exists(self.ehr_file):
            with open(self.ehr_file, 'r') as file:
                data = json.load(file)
            return data
        else:
            return {}

    def get_patient_record(self, patient_id):
        return self.records.get(patient_id, None)

    def update_patient_record(self, patient_id, update_data):
        if patient_id in self.records:
            self.records[patient_id].update(update_data)
        else:
            self.records[patient_id] = update_data
        self.save_records()

    def save_records(self):
        with open(self.ehr_file, 'w') as file:
            json.dump(self.records, file, indent=4)

prescription_manager = PrescriptionManager()
input_processor = InputProcessor()
ml_predictor = MLPredictor()
ehr_manager = EHRManager()

app = FastAPI(title="Medical Records API")

@app.get("/", response_class=HTMLResponse)
async def read_form():
    html_content = """
    <html>
      <head>
        <title>Update Medical Records</title>
      </head>
      <body>
        <h2>Update Your Medical Records</h2>
        <p>
          Step 1 - (Optional) Upload an image/document of your previous medical records.
        </p>
        <p>
          Step 2 - Provide your current readings. You can either type them manually in the text area below 
          or click on <strong>"Record Voice Command"</strong> to capture your command using your device's microphone.
          For example: <em>Diabetes 200, Blood Pressure 150/80, Age 35, 65 kgs, SPO2 98</em>
        </p>
        <form action="/update_record" enctype="multipart/form-data" method="post">
          <label for="image_file">Upload Image (optional):</label><br>
          <input type="file" name="image_file"><br><br>
          <label for="voice_input">Enter Current Readings as Text:</label><br>
          <textarea name="voice_input" id="voice_input" rows="4" cols="50" placeholder="Diabetes 200, Blood Pressure 150/80, Age 35, 65 kgs, SPO2 98"></textarea><br><br>
          <button type="button" id="record-btn">Record Voice Command</button><br><br>
          <input type="hidden" name="patient_id" value="patient_001">
          <input type="submit" value="Update Records">
        </form>
        <script>
          // Check if the browser supports the Web Speech API.
          var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
          if (SpeechRecognition) {
            const recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = false;
            recognition.lang = 'en-US';

            document.getElementById('record-btn').addEventListener('click', function() {
              recognition.start();
            });

            recognition.onresult = function(event) {
              const transcript = event.results[0][0].transcript;
              document.getElementById('voice_input').value = transcript;
            };

            recognition.onerror = function(event) {
              console.error('Speech recognition error', event);
            };
          } else {
            // Disable the record button if not supported.
            document.getElementById('record-btn').disabled = true;
            document.getElementById('record-btn').innerText = 'Voice recording not supported';
          }
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/update_record", response_class=HTMLResponse)
async def update_record(
    patient_id: str = Form("patient_001"),
    voice_input: str = Form(""),
    image_file: UploadFile = File(None)
):
    current_record = ehr_manager.get_patient_record(patient_id)
    if not current_record:
        current_record = {"diabetes": "", "blood_pressure": "", "age": "", "weight": "", "spo2": ""}
        print("No current record found. Creating a new one with empty current readings.")
    else:
        print("Current patient record found:")
        print(current_record)

    if image_file is not None and image_file.filename:
        try:
            contents = await image_file.read()
            suffix = os.path.splitext(image_file.filename)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(contents)
                temp_file_path = tmp.name
            ocr_text = input_processor.perform_ocr(temp_file_path)
            os.remove(temp_file_path)
            print("\nExtracted text from the image:")
            print(ocr_text)
            previous_readings = input_processor.extract_parameters(ocr_text)
            if previous_readings:
                current_record['previous_readings'] = previous_readings
                print("\nExtracted previous readings from the document:")
                print(previous_readings)
            else:
                print("No parameters extracted from the image.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
    else:
        print("No image file provided. Skipping image processing.")

    print("\nProcessing current readings from input text:")
    current_parameters = input_processor.extract_parameters(voice_input)
    if current_parameters:
        print("\nExtracted current readings:")
        print(current_parameters)
        current_record.update(current_parameters)
    else:
        return HTMLResponse(content="<h3>Error: No parameters could be extracted from the input.</h3>", status_code=400)

    prescription_manager.update_prescription(patient_id, current_record)
    risk = ml_predictor.predict_disease_risk(current_record)
    current_record['disease_risk'] = risk
    ehr_manager.update_patient_record(patient_id, current_record)
    result_html = f"""
    <html>
      <head>
        <title>Updated Medical Records</title>
      </head>
      <body>
        <h2>EHR Updated Successfully</h2>
        <pre>{json.dumps(current_record, indent=4)}</pre>
        <p><a href="/">Go Back</a></p>
      </body>
    </html>
    """
    return HTMLResponse(content=result_html)

@app.get("/ehr/{patient_id}", response_class=HTMLResponse)
async def get_ehr(patient_id: str):
    record = ehr_manager.get_patient_record(patient_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Patient record not found.")
    html_content = f"""
    <html>
      <head>
        <title>Patient Record: {patient_id}</title>
      </head>
      <body>
        <h2>Patient Record for {patient_id}</h2>
        <pre>{json.dumps(record, indent=4)}</pre>
        <p><a href="/">Go Back</a></p>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



