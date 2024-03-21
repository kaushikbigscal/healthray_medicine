import json
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
@app.route('/', methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        myDict = request.form
        symptom = myDict['Symptom']
        inputFeatures = [symptom]
        subfeatures = inputFeatures[0]
        symptom_string = subfeatures.strip('[]')
        symptom_list = [symptom.strip('"') for symptom in symptom_string.split(',')]
        print(symptom_list)

        #load main Model file
        file_model = open('main_model.pkl', 'rb')
        model_main = pickle.load(file_model)
        file_model.close()

        # Load LabelEncoders
        with open('main_le_symptom.pkl', 'rb') as f:
            le_Symptom = pickle.load(f)
        with open('main_le_notes.pkl', 'rb') as f:
            le_Notes = pickle.load(f)
        with open('main_le_medicine.pkl', 'rb') as f:
            le_Medicine = pickle.load(f)
        with open('main_le_diagnosis.pkl', 'rb') as f:
            le_Diagnosis = pickle.load(f)

        # Initialize lists to store predictions for each symptom
        predicted_notes_encoded_list = []
        predicted_medicine_encoded_list = []
        predicted_diagnosis_encoded_list = []

        # Loop through each symptom
        for symptom in symptom_list:
            # Encode the symptom
            new_symptom_encoded = le_Symptom.transform([symptom])

            # Predict notes, medicine, and diagnosis for the symptom
            predicted_notes_encoded = model_main.predict([new_symptom_encoded])
            predicted_medicine_encoded = model_main.predict([new_symptom_encoded])
            predicted_diagnosis_encoded = model_main.predict([new_symptom_encoded])

            # Append predictions to lists
            predicted_notes_encoded_list.append(predicted_notes_encoded)
            predicted_medicine_encoded_list.append(predicted_medicine_encoded)
            predicted_diagnosis_encoded_list.append(predicted_diagnosis_encoded)

        # reshape list
        predicted_notes_encoded_list = np.reshape(predicted_notes_encoded_list, (-1,))
        predicted_medicine_encoded_list = np.reshape(predicted_medicine_encoded_list, (-1,))
        predicted_diagnosis_encoded_list = np.reshape(predicted_diagnosis_encoded_list, (-1,))

        # Handle unseen labels
        predicted_notes_encoded_list = np.where(predicted_notes_encoded_list < le_Notes.classes_.shape[0],predicted_notes_encoded_list, 0)
        predicted_medicine_encoded_list = np.where(predicted_medicine_encoded_list < le_Medicine.classes_.shape[0],predicted_medicine_encoded_list, 0)
        predicted_diagnosis_encoded_list = np.where(predicted_diagnosis_encoded_list < le_Diagnosis.classes_.shape[0],predicted_diagnosis_encoded_list, 0)

        # Decode the predictions
        predicted_notes = le_Notes.inverse_transform(predicted_notes_encoded_list)
        predicted_medicine = le_Medicine.inverse_transform(predicted_medicine_encoded_list)
        predicted_diagnosis = le_Diagnosis.inverse_transform(predicted_diagnosis_encoded_list)

        #filtered NaN Value
        filtered_medicine = [x for x in predicted_medicine if not isinstance(x, float) and x == x]
        filtered_notes = [x for x in predicted_notes if not isinstance(x, float) and x == x]
        filtered_diagnosis = [x for x in predicted_diagnosis if x != '()']

        print(filtered_diagnosis)

        # output_diagnosis = [x.strip("()").strip("'") for x in filtered_diagnosis]
        # cleaned_list = [element.replace("'", "").replace('"', '') for element in output_diagnosis]
        # main_list = [item.split(',') for item in cleaned_list]
        # comman_list = [item for sublist in main_list for item in sublist]
        # filtered_list = [item for item in comman_list if item != '']
        # no_extra_list = [element.strip() for element in filtered_list]
        # unique_list = []
        # for item in no_extra_list:
        #     if item not in unique_list:
        #         unique_list.append(item)

        def process_diagnosis(filtered_diagnosis, delimiter=','):
            # Remove parentheses and single quotes
            output_diagnosis = [x.strip("()").strip("'") for x in filtered_diagnosis]
            # Remove single and double quotes
            cleaned_list = [element.replace("'", "").replace('"', '') for element in output_diagnosis]
            # Split by the specified delimiter
            main_list = [item.split(delimiter) for item in cleaned_list]
            # Flatten the list
            comman_list = [item for sublist in main_list for item in sublist]
            # Remove empty strings
            filtered_list = [item for item in comman_list if item != '']
            # Remove extra spaces
            no_extra_list = [element.strip() for element in filtered_list]
            # Remove duplicates
            unique_list = []
            for item in no_extra_list:
                if item not in unique_list:
                    unique_list.append(item)
            return unique_list



        medProb = process_diagnosis(filtered_medicine, delimiter=';')
        notesProb = process_diagnosis(filtered_notes, delimiter=',')
        diagProb = process_diagnosis(filtered_diagnosis, delimiter=',')

        # print(medProb)
        # print(notesProb)
        # print(diagProb)

        return render_template('show.html', med=medProb,note=notesProb,diag=diagProb)
    return render_template('index.html')
# run app
if __name__ == "__main__":
    app.run(host='192.168.1.58',port=5000,debug=True)