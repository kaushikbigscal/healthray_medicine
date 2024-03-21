# Required Library
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Read csv file
df = pd.read_csv('./csv/Doctor_Pradip_Dalwadi.csv')

# Encoding categorical variables
le_Symptom = LabelEncoder()
df['Symptom_encoded'] = le_Symptom.fit_transform(df['Symptom'])

le_Medicine = LabelEncoder()#Encon
df['Medicine_encoded'] = le_Medicine.fit_transform(df['Prescriptions'])

le_Notes = LabelEncoder()
df['Notes_encoded'] = le_Notes.fit_transform(df['Notes'])

le_Diagnosis = LabelEncoder()
df['Diagnosis_encoded'] = le_Diagnosis.fit_transform(df['Diagnosis'])

# Function For Spliting Data set
def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# Calling Function
train, test = data_split(df, 0.3)

# Divide Train and Test data
X_train = train[['Symptom_encoded']].to_numpy()
X_test = test[['Symptom_encoded']].to_numpy()

# For Medicine
Y_train = train[['Medicine_encoded','Notes_encoded','Diagnosis_encoded']].to_numpy()
Y_test = test[['Medicine_encoded','Notes_encoded','Diagnosis_encoded']].to_numpy()

# Model For Medicine
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

model_file = open('main_model.pkl', 'wb')

le_symptom_file = open('main_le_symptom.pkl', 'wb')
le_medicine_file = open('main_le_medicine.pkl', 'wb')
le_notes_file = open('main_le_notes.pkl', 'wb')
le_diagnosis_file = open('main_le_diagnosis.pkl', 'wb')

# dump information to that file
pickle.dump(model, model_file)

pickle.dump(le_Symptom, le_symptom_file)
pickle.dump(le_Medicine, le_medicine_file)
pickle.dump(le_Notes, le_notes_file)
pickle.dump(le_Diagnosis, le_diagnosis_file)

model_file.close()

le_symptom_file.close()
le_medicine_file.close()
le_notes_file.close()
le_diagnosis_file.close()




