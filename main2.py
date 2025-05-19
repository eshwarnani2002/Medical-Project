import json
import re
import os
import numpy as np
import joblib
import speech_recognition as sr
import base64
from sklearn.ensemble import RandomForestClassifier
import easyocr
import tempfile
import io
import matplotlib.pyplot as plt
from datetime import datetime
from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import fitz
import matplotlib.dates as mdates


# ---------------- Utility Functions ----------------

def highlight_parameters(text, parameters):
    mapping = {
        "diabetes": r"(diabetes|blood\s*glucose|fasting\s*blood\s*sugar|FBS|blood\s*sugar)",
        "blood_pressure": r"(blood\s*pressure|BP)",
        "age": r"(age|years?\s*old)",
        "weight": r"(weight|wt|mass)",
        "spo2": r"(spo2|oxygen\s*saturation|s\s*p\s*o\s*2|sop)"
    }
    highlighted_text = text
    for key, regex in mapping.items():
        if key in parameters and parameters[key]:
            pattern = re.compile(
                r'({0}\s*(?:requirement)?\s*(?::|-)?\s*({1}))'.format(regex, re.escape(parameters[key])),
                re.IGNORECASE
            )
            highlighted_text = pattern.sub(r'<span class="highlight">\1</span>', highlighted_text)
    return highlighted_text

def extract_text_from_file(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    reader = easyocr.Reader(['en'], gpu=False)
    if file_ext in ['.png', '.jpg', '.jpeg']:
        return " ".join(reader.readtext(file_path, detail=0))
    elif file_ext == '.pdf':
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        return text
    else:
        raise ValueError("Unsupported file format")

def process_multiple_files(files):
    extracted_data = []
    date_pattern = re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')
    hba1c_pattern = re.compile(r'HbA1c\s*[:\-]?\s*([\d\.]+)%?', re.IGNORECASE)
    fbs_pattern = re.compile(r'FBS\s*[:\-]?\s*([\d\.]+)', re.IGNORECASE)
    # Add other patterns as needed

    for file in files:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        temp_file.write(file.file.read())
        temp_file.close()
        text = extract_text_from_file(temp_file.name)
        os.remove(temp_file.name)

        current_date = None
        for line in text.splitlines():
            date_match = date_pattern.search(line)
            if date_match:
                try:
                    current_date = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                except:
                    current_date = None
            values = {}
            hba1c_match = hba1c_pattern.search(line)
            if hba1c_match:
                values['HbA1c'] = hba1c_match.group(1)
            fbs_match = fbs_pattern.search(line)
            if fbs_match:
                values['FBS'] = fbs_match.group(1)
            # Add other value extractions as needed

            if current_date and values:
                extracted_data.append({"date": current_date, "text": line})
    extracted_data.sort(key=lambda x: x['date'])
    return extracted_data

def extract_diabetes_value(text):
    patterns = {
        "HbA1c": r"HbA1c[:\s]*([\d\.]+)%?",
        "FBS": r"(?:FBS|fasting blood sugar)[:\s]*([\d\.]+)",
        "Blood Glucose": r"(?:blood glucose|blood sugar)[:\s]*([\d\.]+)",
        "Blood Glucose mg/dL": r"(?:blood glucose|blood sugar).*?([\d\.]+)\s*mg/dL"
    }
    extracted_values = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            extracted_values[key] = match.group(1)
    return extracted_values

def generate_diabetes_graph(data):
    plt.figure(figsize=(10, 5))
    plot_dates = []
    plot_values = []
    for entry in data:
        extracted = extract_diabetes_value(entry["text"])
        value = None
        if "FBS" in extracted:
            try:
                value = float(extracted["FBS"])
            except:
                pass
        elif "Blood Glucose" in extracted:
            try:
                value = float(extracted["Blood Glucose"])
            except:
                pass
        elif "Blood Glucose mg/dL" in extracted:
            try:
                value = float(extracted["Blood Glucose mg/dL"])
            except:
                pass
        elif "HbA1c" in extracted:
            try:
                value = float(extracted["HbA1c"])
            except:
                pass
        if value is not None:
            plot_dates.append(entry["date"])
            plot_values.append(value)
    if plot_dates and plot_values:
        plt.plot(plot_dates, plot_values, marker='o', linestyle='-', color='b', label='Blood Glucose Level')
        plt.axhline(y=140, color='g', linestyle='--', label='Threshold (140 mg/dL)')
        plt.xlabel('Date')
        plt.ylabel('Blood Glucose Level (mg/dL)')
        plt.title('Diabetes Trend Over Time')
        plt.legend()
        plt.xticks(rotation=45)
        import matplotlib.dates as mdates
        # Use abbreviated month names (e.g., Jan, Feb, Mar)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    else:
        plt.text(0.5, 0.5, 'No valid blood sugar measurements found', 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=plt.gca().transAxes, fontsize=14)
        plt.axis('off')
    plt.tight_layout()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
def generate_parameter_graph(data, parameter_type):
    """
    Generates a trend bar graph for the given parameter type.
    Supports: 'diabetes', 'hba1c', 'blood_pressure', 'age'.
    - 'age': bar graph showing counts of age >=1 and age 1-18 by year.
    - others: bar graph of measurements over time with threshold line.
    """
    plt.figure(figsize=(10, 5))

    # Special case: age counts (bar graph)
    if parameter_type == 'age':
        counts = {}
        for entry in data:
            year = entry['date'].year
            params = extract_parameter_value(entry['text'], 'age')
            if 'age' not in params:
                continue
            try:
                age_val = int(params['age'])
            except ValueError:
                continue
            counts.setdefault(year, {'1+': 0, '1-18': 0})
            if age_val >= 1:
                counts[year]['1+'] += 1
            if 1 <= age_val <= 18:
                counts[year]['1-18'] += 1

        if not counts:
            plt.text(0.5, 0.5, 'No valid age data found', ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
        else:
            years = sorted(counts)
            labels = [f"{y}/{str(y+1)[-2:]}" for y in years]
            series_all = [counts[y]['1+'] for y in years]
            series_youth = [counts[y]['1-18'] for y in years]

            x = np.arange(len(labels))
            width = 0.35
            plt.bar(x - width/2, series_all, width, color='b', label='Age 1+')
            plt.bar(x + width/2, series_youth, width, color='r', label='Age 1 to 18')

            plt.xlabel('Year')
            plt.ylabel('Count')
            plt.title('Age Distribution Over Time')
            plt.legend()
            plt.xticks(x, labels, rotation=45)
            plt.tight_layout()
        return

    # Prepare data for bar graph (other parameters)
    plot_dates = []
    plot_values = []
    for entry in data:
        extracted = extract_parameter_value(entry['text'], parameter_type)
        value = None
        if parameter_type == 'diabetes':
            for key in ['FBS', 'Blood Glucose', 'Blood Glucose mg/dL', 'HbA1c']:
                if key in extracted:
                    try:
                        value = float(extracted[key])
                        break
                    except:
                        continue
        elif parameter_type == 'hba1c':
            if 'HbA1c' in extracted:
                try:
                    value = float(extracted['HbA1c'])
                except:
                    pass
        elif parameter_type == 'blood_pressure':
            if 'Systolic' in extracted:
                try:
                    value = float(extracted['Systolic'])
                except:
                    pass
        if value is not None:
            plot_dates.append(entry['date'])
            plot_values.append(value)

    if plot_dates and plot_values:
        # Draw bar graph instead of line graph
        x = np.arange(len(plot_dates))
        plt.bar(x, plot_values, color='skyblue', label=f'{parameter_type.replace("_"," ").title()} Level', alpha=0.8)

        # Threshold line and labels
        if parameter_type == 'diabetes':
            threshold = 140
            plt.axhline(threshold, linestyle='--', color='g', label='Threshold (140 mg/dL)')
            ylabel = 'Blood Glucose (mg/dL)'
        elif parameter_type == 'hba1c':
            threshold = 6.5
            plt.axhline(threshold, linestyle='--', color='g', label='Threshold (6.5%)')
            ylabel = 'HbA1c (%)'
        elif parameter_type == 'blood_pressure':
            threshold = 140
            plt.axhline(threshold, linestyle='--', color='g', label='Systolic Threshold (140 mmHg)')
            ylabel = 'Blood Pressure (mmHg)'
        else:
            ylabel = parameter_type.replace('_',' ').title()

        plt.xlabel('Date')
        plt.ylabel(ylabel)
        plt.title(f'{parameter_type.replace("_"," ").title()} Trend Over Time')
        plt.legend()
        date_labels = [d.strftime('%d-%b-%Y') for d in plot_dates]
        plt.xticks(x, date_labels, rotation=45)
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, f'No valid {parameter_type} data found', ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
        plt.tight_layout()

def extract_parameter_value(text, parameter_type):
    patterns = {
        "diabetes": {
            "HbA1c": r"HbA1c\s*[:\-]?\s*([\d\.]+)%?",
            "FBS": r"(?:FBS|fasting blood sugar)\s*[:\-]?\s*([\d\.]+)",
            "Blood Glucose": r"(?:blood glucose|blood sugar)\s*[:\-]?\s*([\d\.]+)",
            "Blood Glucose mg/dL": r"(?:blood glucose|blood sugar).*?([\d\.]+)\s*mg/dL"
        },
        "hba1c": {
            "HbA1c": r"HbA1c\s*[:\-]?\s*([\d\.]+)%?"
        },
        "blood_pressure": {
            "Systolic": r"(?:systolic|systolic blood pressure)\s*[:\-]?\s*([\d\.]+)",
            "Diastolic": r"(?:diastolic|diastolic blood pressure)\s*[:\-]?\s*([\d\.]+)"
        }
    }
    extracted_values = {}
    if parameter_type in patterns:
        for key, pattern in patterns[parameter_type].items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted_values[key] = match.group(1)
    return extracted_values

# ---------------- Classes ----------------

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
        self.recognizer = sr.Recognizer()

    def extract_parameters(self, text):
        parameters = {}
        pattern_dict = {
            'diabetes': [
                r"(?:diabetes|blood\s*glucose|fasting\s*blood\s*sugar|FBS|blood\s*sugar)(?:\s*requirement)?\s*(?::|-)?\s*(\d+|low|normal|high)"
            ],
            'blood_pressure': [
                r"(?:blood\s*pressure|BP)\s*(?::|-)?\s*(\d{2,3})(?:\s*(?:/|by|upon)\s*(\d{2,3}))?"
            ],
            'age': [
                r"age\s*(?::|-)?\s*(\d+)",
                r"(\d+)\s*(?:years old|yrs?)"
            ],
            'weight': [
                r"(?:weight|wt|mass)\s*(?::|-)?\s*(\d+\s*(?:kg|kgs|kilograms)?)"
            ],
            'spo2': [
                r"(?:spo2|oxygen\s*saturation|s\s*p\s*o\s*2|sop)\s*(?::|-)?\s*(\d{2,3})%?"
            ]
        }
        for key, patterns in pattern_dict.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if key == 'blood_pressure':
                        numerator = match.group(1).strip() if match.group(1) else ""
                        denominator = match.group(2).strip() if match.lastindex and match.lastindex >= 2 and match.group(2) else ""
                        parameters[key] = f"{numerator}/{denominator}" if denominator else numerator
                    else:
                        parameters[key] = match.group(1).strip()
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
        bp = bp if bp.strip() else '120/80'
        try:
            systolic = int(bp.split('/')[0]) if '/' in bp else int(bp)
        except ValueError:
            systolic = 120
        spo2_str = parameters.get('spo2', '98')
        spo2_str = spo2_str if spo2_str.strip() else '98'
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

# ---------------- Initialize Managers ----------------

prescription_manager = PrescriptionManager()
input_processor = InputProcessor()
ml_predictor = MLPredictor()
ehr_manager = EHRManager()

# ---------------- FastAPI Application & Endpoints ----------------

app = FastAPI(title="Medical Records API")

@app.get("/", response_class=HTMLResponse)
async def read_form():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <title>Medical Records Portal</title>
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
          body {
              body {
    background: url("https://images.unsplash.com/photo-1506744038136-46273834b3fb?auto=format&fit=crop&w=1600&q=80") no-repeat center center fixed;
    background-size: cover;
          }
          .overlay {
              background-color: rgba(255, 255, 255, 0.95);
              min-height: 100vh;
              padding: 30px;
          }
          .card {
              border-radius: 8px;
              margin-bottom: 20px;
          }
          .highlight {
              background-color: yellow;
              font-weight: bold;
          }
        </style>
      </head>
      <body>
        <div class="overlay">
          <div class="container">
            <h1 class="text-center text-primary mb-4">Medical Records Portal</h1>
            <!-- Section 1: Update Current Record -->
            <div class="card p-4">
              <h3>Update Current Record</h3>
              <form action="/update_record" enctype="multipart/form-data" method="post">
                <div class="form-group">
                  <label>Upload a Recent Medical Record (Image/PDF):</label>
                  <input type="file" class="form-control-file" name="document_file" accept=".png,.jpg,.jpeg,.pdf">
                </div>
                <div class="form-group">
                  <label>Upload a Voice File (MP3 or MP4) for Current Readings:</label>
                  <input type="file" class="form-control-file" name="voice_file" accept="audio/*,video/mp4">
                </div>
                <div class="form-group">
                  <label>Enter Current Readings (or use voice command):</label>
                  <textarea class="form-control" name="voice_input" rows="3" placeholder="E.g.: Diabetes 100, BloodPressure 120 by 40, Age 20, 58 kg, SPO2 98"></textarea>
                </div>
                <input type="hidden" name="patient_id" value="patient_001">
                <div class="form-group">
                  <button type="button" class="btn btn-secondary" id="record-btn">Record Voice Command</button>
                </div>
                <button type="submit" class="btn btn-primary">Update Record</button>
              </form>
            </div>
            <!-- Section 2: Upload Past Reports -->
            <div class="card p-4">
              <h3>Upload Past Reports (Multiple Files)</h3>
              <form action="/upload_files/" enctype="multipart/form-data" method="post">
                <div class="form-group">
                  <label>Select reports (Images/PDFs):</label>
                  <input type="file" class="form-control-file" name="files" accept=".png,.jpg,.jpeg,.pdf" multiple>
                </div>
                <button type="submit" class="btn btn-primary">Upload Past Reports</button>
              </form>
            </div>
            <!-- Section 3: View Specific Parameter Graph -->
            <div class="card p-4">
              <h3>View Trend Graph by Parameter</h3>
              <form action="/show_graph" enctype="multipart/form-data" method="post">
                <div class="form-group">
                  <label>Enter parameter (e.g., Diabetes, HbA1c, Blood Pressure) or use voice command:</label>
                  <textarea class="form-control" name="graph_command" rows="1" placeholder="E.g.: Show me the diabetes graph"></textarea>
                </div>
                <div class="form-group">
                  <label>Upload multiple files for graph data (optional):</label>
                  <input type="file" class="form-control-file" name="files" accept=".png,.jpg,.jpeg,.pdf" multiple>
                </div>
                <div class="form-group">
                  <button type="button" class="btn btn-secondary" id="record-graph-btn">Record Voice Command for Graph</button>
                </div>
                <button type="submit" class="btn btn-info">Show Graph</button>
              </form>
            </div>
          </div>
          <div class="container text-center">
            <a href="/view_diabetes_pattern/" class="btn btn-info">Show Trend Graphically</a>
          </div>
        </div>
        <script>
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
              document.getElementsByName('voice_input')[0].value = transcript;
            };
            recognition.onerror = function(event) {
              console.error('Speech recognition error', event);
            };

            // New listener for the graph command button
            const graphRecognition = new SpeechRecognition();
            graphRecognition.continuous = false;
            graphRecognition.interimResults = false;
            graphRecognition.lang = 'en-US';
            document.getElementById('record-graph-btn').addEventListener('click', function() {
                graphRecognition.start();
            });
            graphRecognition.onresult = function(event) {
                const transcript = event.results[0][0].transcript;
                document.getElementsByName('graph_command')[0].value = transcript;
            };
            graphRecognition.onerror = function(event) {
                console.error('Speech recognition error', event);
            };

          } else {
            document.getElementById('record-btn').disabled = true;
            document.getElementById('record-btn').innerText = 'Voice recording not supported';
            // Disable the new button as well if not supported
            const graphRecordBtn = document.getElementById('record-graph-btn');
            if (graphRecordBtn) {
                graphRecordBtn.disabled = true;
                graphRecordBtn.innerText = 'Voice recording not supported';
            }
          }
        </script>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload_files/")
async def upload_files(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    extracted_data = process_multiple_files(files)
    with open("extracted_data.json", "w") as f:
        json.dump(extracted_data, f, default=str, indent=4)
    generate_diabetes_graph(extracted_data)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    plt.clf()
    diabetes_values_list = []
    for entry in extracted_data:
        values = extract_diabetes_value(entry["text"])
        if values:
            date_obj = entry["date"]
            if isinstance(date_obj, datetime):
                date_str = date_obj.strftime("%Y-%m-%d")
            else:
                date_str = str(date_obj)
            diabetes_values_list.append({"date": date_str, "values": values})
    table_html = "<h3>Extracted Diabetes Values</h3>"
    if diabetes_values_list:
        table_html += "<table class='table table-bordered table-striped'><thead><tr><th>Date</th><th>Values</th></tr></thead><tbody>"
        for d in diabetes_values_list:
            val_str = "<br>".join(f"<strong>{k}:</strong> {v}" for k, v in d["values"].items())
            table_html += f"<tr><td>{d['date']}</td><td>{val_str}</td></tr>"
        table_html += "</tbody></table>"
    else:
        table_html += "<p>No diabetes values extracted from the uploaded reports.</p>"
    html_content = f"""
    <html>
        <head>
            <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
            <style>
              body {{
                background: url("https://source.unsplash.com/1600x900/?hospital,medicine") no-repeat center center fixed;
                background-size: cover;
              }}
              .overlay {{
                background-color: rgba(255, 255, 255, 0.95);
                min-height: 100vh;
                padding: 30px;
              }}
            </style>
        </head>
        <body>
            <div class="overlay">
                <h1>Diabetes Pattern Graph</h1>
                <img src="data:image/png;base64,{encoded_image}" class="img-fluid" alt="Diabetes Graph" />
                {table_html}
                <br><a href="/" class="btn btn-secondary">Go Back</a>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/view_diabetes_pattern/")
async def view_diabetes_pattern():
    try:
        with open("extracted_data.json", "r") as f:
            data = json.load(f)
        extracted_data = []
        for entry in data:
            try:
                dt = datetime.strptime(str(entry["date"])[:10], "%Y-%m-%d")
            except:
                dt = datetime.now()
            extracted_data.append({"date": dt, "text": entry["text"]})
        generate_diabetes_graph(extracted_data)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
        buffer.close()
        plt.clf()
        diabetes_values_list = []
        for entry in extracted_data:
            values = extract_diabetes_value(entry["text"])
            if values:
                diabetes_values_list.append({
                    "date": entry["date"].strftime("%Y-%m-%d"),
                    "values": values
                })
        table_html = "<h3>Extracted Diabetes Values</h3>"
        if diabetes_values_list:
            table_html += "<table class='table table-bordered table-striped'><thead><tr><th>Date</th><th>Values</th></tr></thead><tbody>"
            for d in diabetes_values_list:
                val_str = "<br>".join(f"<strong>{k}:</strong> {v}" for k, v in d["values"].items())
                table_html += f"<tr><td>{d['date']}</td><td>{val_str}</td></tr>"
            table_html += "</tbody></table>"
        else:
            table_html += "<p>No diabetes values extracted from the uploaded reports.</p>"
        html_content = f"""
        <html>
            <head>
                <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
                <style>
                  body {{
                    background: url("https://source.unsplash.com/1600x900/?hospital,medicine") no-repeat center center fixed;
                    background-size: cover;
                  }}
                  .overlay {{
                    background-color: rgba(255, 255, 255, 0.95);
                    min-height: 100vh;
                    padding: 30px;
                  }}
                </style>
            </head>
            <body>
                <div class="overlay">
                    <h1>Diabetes Pattern Graph</h1>
                    <img src="data:image/png;base64,{encoded_image}" class="img-fluid" alt="Diabetes Graph" />
                    {table_html}
                    <br><a href="/" class="btn btn-secondary">Go Back</a>
                </div>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return {"error": "No extracted data found. Please upload past reports first."}

@app.post("/update_record", response_class=HTMLResponse)
async def update_record(
    patient_id: str = Form("patient_001"),
    voice_input: str = Form(""),
    document_file: UploadFile = File(None),
    voice_file: UploadFile = File(None)
):
    current_record = ehr_manager.get_patient_record(patient_id)
    if not current_record:
        current_record = {"diabetes": "", "blood_pressure": "", "age": "", "weight": "", "spo2": ""}
        print("No current record found. Creating a new one with empty current readings.")
    else:
        print("Current patient record found:")
        print(current_record)
    previous_ocr_text = ""
    previous_parameters = {}
    if document_file is not None and document_file.filename:
        try:
            contents = await document_file.read()
            suffix = os.path.splitext(document_file.filename)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(contents)
                temp_file_path = tmp.name
            if suffix in [".png", ".jpg", ".jpeg"]:
                ocr_text = input_processor.perform_ocr(temp_file_path)
            elif suffix == ".pdf":
                try:
                    doc = fitz.open(temp_file_path)
                    ocr_text = ""
                    for i in range(doc.page_count):
                        page = doc.load_page(i)
                        pix = page.get_pixmap()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                            tmp_img.write(pix.tobytes("png"))
                            temp_img_path = tmp_img.name
                        text = input_processor.perform_ocr(temp_img_path)
                        ocr_text += text + "\n"
                        os.remove(temp_img_path)
                    doc.close()
                except Exception as e:
                    ocr_text = f"Error processing PDF with PyMuPDF: {e}"
            else:
                ocr_text = "Unsupported file format for document."
            os.remove(temp_file_path)
            print("\nExtracted text from the document:")
            print(ocr_text)
            previous_parameters = input_processor.extract_parameters(ocr_text)
            if previous_parameters:
                current_record['previous_readings'] = previous_parameters
            previous_ocr_text = ocr_text
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing document: {e}")
    else:
        print("No document file provided. Skipping document processing.")
    voice_transcript = ""
    if voice_file is not None and voice_file.filename:
        try:
            voice_contents = await voice_file.read()
            suffix = os.path.splitext(voice_file.filename)[1].lower()
            if suffix == ".mp4":
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_video:
                    tmp_video.write(voice_contents)
                    temp_video_path = tmp_video.name
                try:
                    from moviepy.editor import AudioFileClip
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                        temp_wav_path = tmp_wav.name
                    clip = AudioFileClip(temp_video_path)
                    clip.write_audiofile(temp_wav_path, logger=None)
                    clip.close()
                except Exception as e:
                    print("Error converting MP4 to WAV:", e)
                    temp_wav_path = None
                os.remove(temp_video_path)
                voice_processing_path = temp_wav_path
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_voice:
                    tmp_voice.write(voice_contents)
                    voice_processing_path = tmp_voice.name
            if voice_processing_path:
                try:
                    r = sr.Recognizer()
                    with sr.AudioFile(voice_processing_path) as source:
                        audio_data = r.record(source)
                        voice_transcript = r.recognize_google(audio_data)
                        print("Voice file transcript:", voice_transcript)
                except Exception as e:
                    print("Error processing voice file:", e)
                    voice_transcript = ""
                os.remove(voice_processing_path)
            if voice_transcript:
                voice_input = voice_transcript
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing voice file: {e}")
    else:
        print("No voice file provided. Using voice input from the text area.")
    print("\nProcessing current readings from input text:")
    current_parameters = input_processor.extract_parameters(voice_input)
    if current_parameters:
        print("\nExtracted current readings:")
        print(current_parameters)
        current_record.update(current_parameters)
    else:
        return HTMLResponse(content="<h3>Error: No parameters could be extracted from the input/voice data.</h3>", status_code=400)
    prescription_manager.update_prescription(patient_id, current_record)
    risk = ml_predictor.predict_disease_risk(current_record)
    current_record['disease_risk'] = risk
    ehr_manager.update_patient_record(patient_id, current_record)
    table_rows = ""
    for key, value in current_record.items():
        if isinstance(value, dict):
            inner_rows = ""
            for subkey, subvalue in value.items():
                inner_rows += f"<tr><td>{subkey.capitalize()}</td><td>{subvalue}</td></tr>"
            value = f"<table class='table table-sm table-bordered'>{inner_rows}</table>"
        table_rows += f"<tr><td>{key.replace('_', ' ').capitalize()}</td><td>{value}</td></tr>"
    highlighted_text = ""
    if previous_ocr_text and previous_parameters:
        highlighted_text = highlight_parameters(previous_ocr_text, previous_parameters)
    result_html = f"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <title>Updated Medical Records</title>
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
          body {{
              background: url("https://source.unsplash.com/1600x900/?hospital,medicine") no-repeat center center fixed;
              background-size: cover;
          }}
          .overlay {{
              background-color: rgba(255, 255, 255, 0.95);
              min-height: 100vh;
              padding: 30px;
          }}
          .card {{
              border-radius: 8px;
          }}
          .highlight {{
              background-color: yellow;
              font-weight: bold;
          }}
        </style>
      </head>
      <body>
        <div class="overlay">
          <div class="container">
            <div class="card p-4 mb-4">
              <h2 class="text-primary">EHR Updated Successfully</h2>
              <h4>Updated Record</h4>
              <table class="table table-bordered">
                <thead class="thead-dark">
                  <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {table_rows}
                </tbody>
              </table>
            </div>
    """
    # Add graph generation based on voice command from the update section
    graph_html = ""
    if voice_transcript:
        # Check for graph-related commands
        graph_commands = {
            "diabetes": ["diabetes", "blood sugar", "glucose"],
            "hba1c": ["hba1c", "a1c", "glycated hemoglobin"],
            "blood_pressure": ["blood pressure", "bp", "systolic", "diastolic"]
        }

        command_found = False
        for param_type, keywords in graph_commands.items():
            if any(keyword in voice_transcript.lower() for keyword in keywords):
                try:
                    with open("extracted_data.json", "r") as f:
                        data = json.load(f)
                    extracted_data = []
                    for entry in data:
                        try:
                            dt = datetime.strptime(str(entry["date"])[:10], "%Y-%m-%d")
                        except:
                            dt = datetime.now()
                        extracted_data.append({"date": dt, "text": entry["text"]})

                    generate_parameter_graph(extracted_data, param_type)
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format="png")
                    buffer.seek(0)
                    encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
                    buffer.close()
                    plt.clf()

                    graph_html = f"""
                    <div class="card p-4 mb-4">
                        <h4>{param_type.replace('_', ' ').title()} Trend Graph</h4>
                        <img src="data:image/png;base64,{encoded_image}" class="img-fluid" alt="{param_type} Graph" />
                    </div>
                    """
                    command_found = True
                    break
                except FileNotFoundError:
                    graph_html = f"""
                    <div class="card p-4 mb-4">
                        <p>No historical data found to generate the {param_type.replace('_', ' ')} graph. Please upload past reports.</p>
                    </div>
                    """
                    command_found = True # Indicate that a graph command was handled
                    break
                except Exception as e:
                    graph_html = f"""
                    <div class="card p-4 mb-4">
                        <p>Error generating {param_type.replace('_', ' ')} graph: {e}</p>
                    </div>
                    """
                    command_found = True # Indicate that a graph command was handled
                    break


    result_html += graph_html

    if highlighted_text:
        result_html += f"""
          <div class="card p-4 mb-4">
            <h4>Extracted Text from Document (highlighted parameters):</h4>
            <pre>{highlighted_text}</pre>
          </div>
        """
    if voice_transcript:
        result_html += f"""
          <div class="card p-4 mb-4">
            <h4>Voice File Transcript:</h4>
            <pre>{voice_transcript}</pre>
          </div>
        """
    result_html += """
            <a href="/" class="btn btn-secondary">Go Back</a>
          </div>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=result_html)

@app.get("/ehr/{patient_id}", response_class=HTMLResponse)
async def get_ehr(patient_id: str):
    record = ehr_manager.get_patient_record(patient_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Patient record not found.")
    table_rows = ""
    for key, value in record.items():
        if isinstance(value, dict):
            inner_rows = ""
            for subkey, subvalue in value.items():
                inner_rows += f"<tr><td>{subkey.capitalize()}</td><td>{subvalue}</td></tr>"
            value = f"<table class='table table-sm table-bordered'>{inner_rows}</table>"
        table_rows += f"<tr><td>{key.replace('_', ' ').capitalize()}</td><td>{value}</td></tr>"
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="UTF-8">
        <title>Patient Record: {patient_id}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
        <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
        <style>
          body {{
              background: url("https://source.unsplash.com/1600x900/?hospital,medicine") no-repeat center center fixed;
              background-size: cover;
          }}
          .overlay {{
              background-color: rgba(255, 255, 255, 0.95);
              min-height: 100vh;
              padding: 30px;
          }}
          .card {{
              border-radius: 8px;
          }}
        </style>
      </head>
      <body>
        <div class="overlay">
          <div class="container">
            <div class="card p-4">
              <h2 class="text-primary">Patient Record for {patient_id}</h2>
              <table class="table table-bordered">
                <thead class="thead-dark">
                  <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {table_rows}
                </tbody>
              </table>
              <a href="/" class="btn btn-secondary">Go Back</a>
            </div>
          </div>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/show_graph", response_class=HTMLResponse)
async def show_graph(
    graph_command: str = Form(""),
    files: list[UploadFile] = File(None)
):
    if not graph_command and not files:
        return HTMLResponse(content="<h3>Please provide a parameter or upload files to generate a graph.</h3>", status_code=400)

    extracted_data = []
    if files:
        extracted_data = process_multiple_files(files)
    elif os.path.exists("extracted_data.json"):
        try:
            with open("extracted_data.json", "r") as f:
                data_from_json = json.load(f)
                for entry in data_from_json:
                    try:
                        dt = datetime.strptime(str(entry["date"])[:10], "%Y-%m-%d")
                    except:
                        dt = datetime.now()
                    extracted_data.append({"date": dt, "text": entry["text"]})
        except Exception as e:
            print(f"Error loading data from extracted_data.json: {e}")
            # Continue even if loading fails, maybe new files were uploaded

    if not extracted_data:
         return HTMLResponse(content="<h3>No data available to generate a graph. Please upload reports first.</h3>", status_code=400)


    # Determine which graph is requested
    graph_commands = {
        "diabetes": ["diabetes", "blood sugar", "glucose"],
        "hba1c": ["hba1c", "a1c", "glycated hemoglobin"],
        "blood_pressure": ["blood pressure", "bp", "systolic", "diastolic"]
    }

    requested_param_type = None
    for param_type, keywords in graph_commands.items():
        if any(keyword in graph_command.lower() for keyword in keywords):
            requested_param_type = param_type
            break

    if not requested_param_type:
        return HTMLResponse(content=f"<h3>Could not understand the request for '{graph_command}'. Please specify Diabetes, HbA1c, or Blood Pressure.</h3>", status_code=400)

    try:
        generate_parameter_graph(extracted_data, requested_param_type)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.read()).decode("utf-8")
        buffer.close()
        plt.clf()

        graph_html = f"""
        <div class="card p-4 mb-4">
            <h4>{requested_param_type.replace('_', ' ').title()} Trend Graph</h4>
            <img src="data:image/png;base64,{encoded_image}" class="img-fluid" alt="{requested_param_type} Graph" />
        </div>
        """

        html_content = f"""
        <html>
            <head>
                <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
                <style>
                  body {{
                    background: url("https://source.unsplash.com/1600x900/?hospital,medicine") no-repeat center center fixed;\n                    background-size: cover;\n                  }}\n                  .overlay {{\n                    background-color: rgba(255, 255, 255, 0.95);\n                    min-height: 100vh;\n                    padding: 30px;\n                  }}\n                </style>\n            </head>\n            <body>\n                <div class="overlay">\n                    <h1>Requested Parameter Graph</h1>\n                    {graph_html}\n                    <br><a href="/" class="btn btn-secondary">Go Back</a>\n                </div>\n            </body>\n        </html>\n        """
        return HTMLResponse(content=html_content)

    except Exception as e:
        return HTMLResponse(content=f"<h3>Error generating graph: {e}</h3>", status_code=500)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
