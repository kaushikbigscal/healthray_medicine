# preprocessing.py
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the LabelEncoders
file_symptom = open('main_le_symptom.pkl', 'rb')
file_medicine = open('main_le_medicine.pkl', 'rb')
file_notes = open('main_le_notes.pkl', 'rb')
file_diagnosis = open('main_le_diagnosis.pkl', 'rb')

le_Symptom = pickle.load(file_symptom)
le_Medicine = pickle.load(file_medicine)
le_Notes = pickle.load(file_notes)
le_Diagnosis = pickle.load(file_diagnosis)

file_symptom.close()
file_medicine.close()
file_notes.close()
file_diagnosis.close()

def preprocess_symptom(symptom):
    return le_Symptom.transform([symptom])

def inverse_transform_notes(predictions):
    return le_Notes.inverse_transform(predictions)

def inverse_transform_medicine(predictions):
    return le_Medicine.inverse_transform(predictions)

def inverse_transform_diagnosis(predictions):
    return le_Diagnosis.inverse_transform(predictions)



def load_models():
    # Load the trained models
    file_model = open('main_model.pkl', 'rb')

    model_main = pickle.load(file_model)

    file_model.close()

    return model_main
