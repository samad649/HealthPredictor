import pandas as pd
import boto3
import io

class DataSet:
    def __init__(self):
        ACCESS_KEY = 'Enter Key Here'
        SECRET_KEY = 'Enter Secret Key here'
        session = boto3.Session(
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY)
        # Initializing file paths for the MIMIC-III data
        self.s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
        self.bucket_name = 'mimic-iv-dataset'
        self.data_files = {
            "admissions_df": "ADMISSIONS.csv.gz",
            "diagnosis_codes_df": "D_ICD_DIAGNOSES.csv.gz",
            "diagnosis_df": "DIAGNOSES_ICD.csv.gz",
            "notes_df": "NOTEEVENTS.csv.gz",
            "patients_df": "PATIENTS.csv.gz",
            "prescription_df": "PRESCRIPTIONS.csv.gz"
        }
        # Dictionary to hold DataFrames
        self.dataframes = {}

    def load_data(self, nrows=1000):
        # Load each CSV file from S3 into a DataFrame
        for df_name, file_key in self.data_files.items():
            try:
                # Check if the current file is the diagnosis file
                if 'diagnosis_codes_df'== df_name:  # Use a more specific check based on the file name
                    obj = self.s3.get_object(Bucket=self.bucket_name, Key=file_key)
                    self.dataframes[df_name] = pd.read_csv(
                        io.BytesIO(obj['Body'].read()),
                        compression='gzip'
                    )
                    print(f"{df_name} loaded successfully with full rows.")
                else:
                    # For other files, load only the first nrows
                    obj = self.s3.get_object(Bucket=self.bucket_name, Key=file_key)
                    self.dataframes[df_name] = pd.read_csv(
                        io.BytesIO(obj['Body'].read()),
                        compression='gzip',
                        nrows=nrows
                    )
                    print(f"{df_name} loaded successfully with {nrows} rows.")
            except Exception as e:
                print(f"Error loading {file_key}: {e}")

    def get_dataframe(self, df_name):
        # Return a specific DataFrame by name
        return self.dataframes.get(df_name, None)

    def print_head(self):
        # Print the first 5 rows of each DataFrame
        for df_name, df in self.dataframes.items():
            if df is not None:
                print(f"\nDataFrame: {df_name}")
                print(df.head())
            else:
                print(f"\nDataFrame: {df_name} could not be loaded.")

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
class NaiveBayes:
    #holds all the main data for model
    df = pd.DataFrame()
    #map that connects diag codes to full name diagnosis
    diagnosis_map = {}
    def __init__(self):
        self.counter = CountVectorizer()

        # Use OneVsRestClassifier to handle multi-label classification
        self.model = OneVsRestClassifier(MultinomialNB())

        self.data = DataSet()

        self.data.load_data(nrows=75000)

        self.mlb = MultiLabelBinarizer()
    # Extract all the needed data and store in df
    def process_data(self):
        admissions = self.data.dataframes["admissions_df"]
        self.df['id'] = admissions['subject_id']

        notes = self.data.dataframes["notes_df"]
        # Get every text from notes_df associated with the subject id
        id_notes = {}
        for _, row in notes.iterrows():
            subject_id = row['subject_id']
            text = row['text']

            if subject_id not in id_notes:
                id_notes[subject_id] = text
            else:
                id_notes[subject_id] += ' ' + text
        # Store each set of notes to its id
        self.df['notes'] = self.df['id'].map(id_notes)

        # Store all the diagnosis with each subject id
        diagnosis = self.data.dataframes["diagnosis_df"]
        id_diagnosis = {}

        for _, row in diagnosis.iterrows():
            subject_id = row['subject_id']
            diag_code = row['icd9_code']

            if subject_id not in id_diagnosis:
                id_diagnosis[subject_id] = [diag_code]
            else:
                id_diagnosis[subject_id].append(diag_code)

        self.df['diagnosis_codes'] = self.df['id'].map(id_diagnosis)

        diagnosis_codes = self.data.dataframes["diagnosis_codes_df"]
        for _, row in diagnosis_codes.iterrows():
            
            code = str(row['icd9_code'])  
            long_title = row['long_title']
            self.diagnosis_map[code] = long_title

        # Print the first 5 items in the diagnosis_map
        print(self.df.head())
        for idx, (key, value) in enumerate(self.diagnosis_map.items()):
            if idx == 5:  # Stop after printing 5 items
                break
            print(f"{key}: {value}")

    def train(self):
        # Drop rows where 'notes' or 'diagnosis_codes' are NaN
        self.df.dropna(subset=['notes', 'diagnosis_codes'], inplace=True)

        # Replace non-list entries with an empty list
        self.df['diagnosis_codes'] = self.df['diagnosis_codes'].apply(
            lambda x: x if isinstance(x, list) else []
        )

        # Check if there are still any NaN values
        if self.df['diagnosis_codes'].isna().sum() > 0:
            print("Warning: There are still NaN values in 'diagnosis_codes' after processing.")

        X = self.df['notes']
        y = self.mlb.fit_transform(self.df['diagnosis_codes'])

        # Train-test split ensuring all classes are represented
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

        X_train_vectorized = self.counter.fit_transform(X_train)
        X_test_vectorized = self.counter.transform(X_test)

        print(X_train_vectorized.shape)
        print(X_test_vectorized.shape)
        print(y_train.shape)
        print(y_test.shape)

        self.model.fit(X_train_vectorized, y_train)

        y_pred = self.model.predict(X_test_vectorized)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))

    def predict(self, X):
        # Vectorize the input text
        X_vectorized = self.counter.transform(X)

        # Get the probabilities for each class (diagnosis)
        probas = self.model.predict_proba(X_vectorized)

        # Get the corresponding diagnosis codes for each class
        diagnosis_codes = self.mlb.classes_

        # For each prediction, get the top 5 diagnoses
        top_5_diagnoses = []
        for prob in probas:
            # Get the indices of the top 5 probabilities
            top_5_indices = np.argsort(prob)[::-1][:5]
            # Map the indices to the actual diagnosis codes
            top_5_codes = diagnosis_codes[top_5_indices]

            # Convert the diagnosis codes to their long title names using the diagnosis_map
            top_5_titles = [self.diagnosis_map.get(str(code), "Unknown Diagnosis") for code in top_5_codes]
            
            top_5_diagnoses.append(top_5_titles)

        return top_5_diagnoses