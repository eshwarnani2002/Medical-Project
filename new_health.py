import json
import re
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
import easyocr
import speech_recognition as sr


# Attempt to import speech_recognition, but handle the ImportError
try:
    import speech_recognition as sr
    speech_recognition_available = True
except ImportError:
    print("speech_recognition library not found. Voice input will be disabled.")
    speech_recognition_available = False

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
        self.use_voice = speech_recognition_available # Use global flag
        if self.use_voice:
            self.recognizer = sr.Recognizer()

    def get_input(self):
        if self.use_voice:
            try:
                with sr.Microphone() as source:
                    print("Listening for voice input... (Please speak clearly)")
                    audio = self.recognizer.listen(source, timeout=5)
                try:
                    text = self.recognizer.recognize_google(audio)
                    print("Voice recognized:", text)
                    return text
                except sr.UnknownValueError:
                    print("Speech Recognition could not understand audio")
                    return input("Please type the parameter values: ")
                except sr.RequestError as e:
                    print(f"Could not request results from Speech Recognition service; {e}")
                    return input("Please type the parameter values: ")
            except Exception as mic_err:
                print("Error accessing the microphone:", mic_err)
                self.use_voice = False # Disable voice
                return input("Microphone not available. Please type the parameter values: ")
        else:
            return input("Enter new parameter values: ")

    def extract_parameters(self, text):
        parameters = {}

        pattern_dict = {
            'diabetes': [
                r"(?:diabetes|blood sugar)\s*(?::|-)?\s*(\d+|low|normal|high)"
            ],
            'blood_pressure': [
                r"blood pressure\s*(?::|-)?\s*([0-9]{2,3}(?:/[0-9]{2,3})?)"
            ],
            'age': [
                r"age\s*(?::|-)?\s*(\d+)",
                r"(\d+)\s*(?:years old|yrs?)"
            ],
            'weight': [
                r"(?:weight|wt)\s*(?::|-)?\s*(\d+\s*(?:kg|kgs|kilograms)?)",
                r"(\d+\s*(?:kg|kgs|kilograms))"
            ],
            'spo2': [
                r"(?:SPO2|SOP)\s*(?::|-)?\s*(\d{2,3})%?",
                r"(\d{2,3})\s*%"
            ]
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

def main():
    prescription_manager = PrescriptionManager()
    input_processor = InputProcessor()
    ml_predictor = MLPredictor()
    ehr_manager = EHRManager()

    patient_id = "patient_001"
    current_record = ehr_manager.get_patient_record(patient_id)
    if not current_record:
        current_record = {
            "diabetes": "",
            "blood_pressure": "",
            "age": "",
            "weight": "",
            "spo2": ""
        }
        print("No current record found. Creating a new one with empty current readings.")
    else:
        print("Current patient record found:")
        print(current_record)

    # Step 1: Process image/document to extract previous readings
    image_path = input("\nStep 1 - Provide the file path for the image/document of your previous medical records: ")
    if os.path.exists(image_path):
        try:
            ocr_text = input_processor.perform_ocr(image_path)
            print("\nExtracted text from the image:")
            print(ocr_text)

            previous_readings = input_processor.extract_parameters(ocr_text)
            if previous_readings:
                current_record['previous_readings'] = previous_readings
                print("\nExtracted previous readings from the uploaded document:")
                print(previous_readings)
            else:
                print("No parameters could be extracted from the image.")
        except Exception as e:
            print("Error processing the image:", e)
    else:
        print("Image file not found. Skipping previous readings extraction.")

    # Step 2: Get current readings via voice input
    print("\nStep 2 - Please provide your current readings (voice or text).")
    voice_input_text = input_processor.get_input() #Now gets input from microphone
    if voice_input_text:
        current_parameters = input_processor.extract_parameters(voice_input_text)
        if current_parameters:
            print("\nExtracted current readings:")
            print(current_parameters)

            current_record.update(current_parameters)
        else:
            print("No parameters could be extracted from the voice/text input.")
    else:
        print("No voice input received. Exiting.")
        return

    prescription_manager.update_prescription(patient_id, current_record)

    risk = ml_predictor.predict_disease_risk(current_record)
    current_record['disease_risk'] = risk

    ehr_manager.update_patient_record(patient_id, current_record)
    print("\nEHR updated successfully. Current patient record:")
    print(ehr_manager.get_patient_record(patient_id))

if __name__ == '__main__':
    main()