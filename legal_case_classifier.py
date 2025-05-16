import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import pickle
import warnings

warnings.filterwarnings('ignore')

nltk_data_path = "/tmp/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
os.environ["NLTK_DATA"] = nltk_data_path

# Download necessary corpora to that path
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)

class LegalCaseClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.legal_terms = {
            'plaintiff', 'defendant', 'appeal', 'judgment', 'court', 'section',
            'act', 'article', 'respondent', 'appellant', 'petitioner', 'accused',
            'evidence', 'conviction', 'forensic', 'witness', 'testimony', 'murder',
            'ipc', 'dismissed', 'upheld', 'affirmed', 'ballistic', 'circumstantial'
        }

    def preprocess_text(self, text):
        """Enhanced legal text preprocessing"""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)  # Keep only letters and whitespace
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace


        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words or word in self.legal_terms]

        # Enhanced lemmatization
        tokens = [self.lemmatizer.lemmatize(word, pos='v') for word in tokens]  # Verbs
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]  # Nouns

        return ' '.join(tokens)

    def load_data(self, file_path):
        try:
            df = pd.read_csv(file_path)
            required_cols = ['case_data', 'case_type', 'judgement']

            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Dataset must contain: {required_cols}")

            df = df.dropna(subset=required_cols)
            df = df[df['case_data'].str.strip().astype(bool)]
            df = df[df['judgement'].str.strip().astype(bool)]

            if len(df) < 5:
                raise ValueError("Minimum 5 samples required")

            return df
        except Exception as e:
            raise ValueError(f"Data loading failed: {str(e)}")

    def analyze_dataset(self, df):

        analysis = {
            'total_cases': len(df),
            'case_types': df['case_type'].value_counts().to_dict(),
            'judgement_distribution': df['judgement'].value_counts().to_dict(),
            'min_samples_per_class': df['judgement'].value_counts().min()
        }

        if len(analysis['judgement_distribution']) < 2:
            raise ValueError("At least 2 judgement classes required")

        return analysis

    def prepare_data(self, df):

        df['legal_features'] = (
                df['case_type'].str.lower() + " " +
                df['case_type'].str.lower() + " " +
                df['case_data']
        )


        df['processed_text'] = df['legal_features'].apply(self.preprocess_text)


        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(df['judgement'])

        test_size = min(0.2, 3/len(df))  # Ensure at least 3 test samples
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], y,
            test_size=test_size,
            random_state=42
        )

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),  # Capture legal phrases
            stop_words=list(self.stop_words - self.legal_terms),
            min_df=2,
            max_df=0.95
        )

        models = [
            {
                'name': 'Random Forest',
                'model': RandomForestClassifier(
                    n_estimators=200,
                    class_weight='balanced',
                    max_depth=None,
                    min_samples_split=5,
                    random_state=42
                )
            },

        ]

        best_score = 0
        best_model = None

        for model_config in models:
            print(f"\nTraining {model_config['name']}...")

            pipeline = Pipeline([
                ('vectorizer', vectorizer),
                ('classifier', model_config['model'])
            ])

            # Simple training without CV for small datasets
            pipeline.fit(X_train, y_train)

            # Evaluate on training data
            y_pred = pipeline.predict(X_train)
            score = f1_score(y_train, y_pred, average='weighted')
            print(f"Training F1 score: {score:.3f}")

            if score > best_score:
                best_score = score
                best_model = pipeline
                print("New best model!")

        return best_model

    def evaluate(self, model, X_test, y_test):
        if len(X_test) == 0:
            print("Warning: No test samples available")
            return {'accuracy': 0.0}

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        present = sorted(set(y_test) | set(y_pred))

        target_names = [self.label_encoder.inverse_transform([i])[0] for i in present]

        print(f"\nTest Accuracy: {accuracy:.4f}\n")
        print("Classification Report (filtered to present classes):")
        report = classification_report(
            y_test,
            y_pred,
            labels=present,
            target_names=target_names,
            zero_division=0
        )
        print(report)

        return {'accuracy': accuracy, 'report': report}

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'preprocessor': self.preprocess_text
            }, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            saved = pickle.load(f)
            self.model = saved['model']
            self.label_encoder = saved['label_encoder']
            self.preprocess_text = saved.get('preprocessor', self.preprocess_text)

    def predict_judgement(self, case_data, case_type):
        if not self.model:
            raise ValueError("Model not trained or loaded")
        legal_input = f"{case_type.lower()} {case_type.lower()} {case_data}"
        processed_input = self.preprocess_text(legal_input)

        pred = self.model.predict([processed_input])[0]
        return self.label_encoder.inverse_transform([pred])[0]

def main():
    try:
        print("=== Legal Case Classification System ===")
        classifier = LegalCaseClassifier()

        print("\nLoading legal case data...")
        df = classifier.load_data("property_disputes.csv")

        print("\nAnalyzing dataset...")
        analysis = classifier.analyze_dataset(df)
        print(f"Total cases: {analysis['total_cases']}")
        print(f"Case types: {analysis['case_types']}")
        print(f"Judgement distribution: {analysis['judgement_distribution']}")
        print(f"Minimum samples per class: {analysis['min_samples_per_class']}")

        print("\nPreparing legal case data...")
        X_train, X_test, y_train, y_test = classifier.prepare_data(df)
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")

        print("\nTraining legal case classifier...")
        classifier.model = classifier.train_model(X_train, y_train)

        print("\nEvaluating model...")
        metrics = classifier.evaluate(classifier.model, X_test, y_test)

        classifier.save_model("legal_case_classifier3.pkl")
        print("\nModel saved successfully!")

        print("\nTesting with sample case...")
        sample_pred = classifier.predict_judgement(
            "The defendant was found with murder weapon matching ballistic evidence",
            "Murder"
        )
        print(f"Predicted judgement: {sample_pred}")

    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()