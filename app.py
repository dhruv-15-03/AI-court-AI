from flask import Flask, request, jsonify
import dill
import os
import sys
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from legal_case_classifier import LegalCaseClassifier
app = Flask(__name__)
CORS(app)

MODEL_PATH = "legal_case_classifier.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Train it first.")

with open(MODEL_PATH, "rb") as f:
    saved = dill.load(f)
    classifier_model = saved["model"]
    label_encoder = saved["label_encoder"]
    preprocess_fn = saved["preprocessor"]

CASE_TYPES = {
    "initial": [
        {"id": "case_type", "question": "What type of case is this?", "options": ["Criminal", "Civil", "Family", "Labor"]}
    ],
    "Criminal": [
        {"id": "parties", "question": "Who are the parties involved in the case?"},
        {"id": "violence_level", "question": "What was the level of violence involved?",
         "options": ["None", "Threat only", "Minor injury", "Serious injury", "Death"]},
        {"id": "weapon", "question": "Was any weapon used?", "options": ["Yes", "No"]},
        {"id": "police_report", "question": "Has a police report been filed?", "options": ["Yes", "No"]},
        {"id": "witnesses", "question": "Are there any witnesses?", "options": ["Yes", "No"]},
        {"id": "premeditation", "question": "Was the act premeditated or spontaneous?",
         "options": ["Premeditated", "Spontaneous", "Unclear"]}
    ],
    "Family": [
        {"id": "parties", "question": "Who are the parties involved in the dispute?"},
        {"id": "marriage_duration", "question": "How long have the parties been married? (if applicable)"},
        {"id": "children", "question": "Are there children involved?", "options": ["Yes", "No"]},
        {"id": "property", "question": "Is there shared property in dispute?", "options": ["Yes", "No"]},
        {"id": "previous_agreements", "question": "Are there any previous agreements between parties?", "options": ["Yes", "No"]},
        {"id": "violence", "question": "Has there been any domestic violence?", "options": ["Yes", "No"]}
    ],
    "Civil": [
        {"id": "parties", "question": "Who are the parties involved in the case?"},
        {"id": "dispute_type", "question": "What is the nature of the dispute?",
         "options": ["Contract", "Property", "Debt", "Damages", "Other"]},
        {"id": "document_evidence", "question": "Is there documentary evidence?", "options": ["Yes", "No"]},
        {"id": "monetary_value", "question": "What is the approximate monetary value of the dispute?"},
        {"id": "prior_relationship", "question": "What was the prior relationship between parties?"},
        {"id": "attempts_resolution", "question": "Have there been previous attempts at resolution?", "options": ["Yes", "No"]}
    ],
    "Labor": [
        {"id": "parties", "question": "Who are the involved parties? (employer/employee)"},
        {"id": "employment_duration", "question": "How long was the employment period?"},
        {"id": "contract_type", "question": "What type of employment contract was in place?",
         "options": ["Full-time", "Part-time", "Contract", "None", "Other"]},
        {"id": "dispute_reason", "question": "What is the main reason for the dispute?",
         "options": ["Termination", "Wages", "Working conditions", "Discrimination", "Other"]},
        {"id": "union_involvement", "question": "Is there union involvement?", "options": ["Yes", "No"]},
        {"id": "previous_complaints", "question": "Were there previous formal complaints?", "options": ["Yes", "No"]}
    ]
}

@app.route("/api/questions", methods=["GET"])
def get_questions():
    return jsonify(CASE_TYPES)

@app.route("/api/questions/<case_type>", methods=["GET"])
def get_questions_by_type(case_type):
    if case_type in CASE_TYPES:
        return jsonify(CASE_TYPES[case_type])
    return jsonify({"error": "Case type not found"}), 404

@app.route("/api/analyze", methods=["POST"])
def analyze_case():
    data = request.json

    combined_text = ""
    for key, value in data.items():
        if key != "case_type":
            combined_text += f"{key}: {value}. "

    processed = preprocess_fn(combined_text)
    pred = classifier_model.predict([processed])[0]
    judgment = label_encoder.inverse_transform([pred])[0]

    case_type = data.get("case_type", "Unknown")
    if case_type == "Unknown":
        if data.get("violence_level", "None") != "None" or data.get("police_report") == "Yes":
            case_type = "Criminal"
        elif data.get("employment_duration"):
            case_type = "Labor"
        elif data.get("children") or data.get("marriage_duration"):
            case_type = "Family"
        else:
            case_type = "Civil"

    return jsonify({
        "case_type": case_type,
        "judgment": judgment,
        "answers": data
    })

if __name__ == "__main__":
    app.run(debug=True,port=5002, host='0.0.0.0')