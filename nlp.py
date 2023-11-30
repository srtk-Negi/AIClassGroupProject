import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import cross_val_score
import nltk
import joblib


def main():
    df = pd.read_csv('spam_ham_dataset.csv')

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    nltk.download('stopwords')

    model = make_pipeline(
        TfidfVectorizer(stop_words=stopwords.words('english'),
                        lowercase=True,
                        strip_accents='ascii',
                        token_pattern=r'\b\w\w+\b'),
        SVC(kernel='linear')
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, 'spam_classifier_model.joblib')

    # Load the model
    loaded_model = joblib.load('spam_classifier_model.joblib')

    # Example of classifying a new email
    new_email = "This is a new email that you want to classify as spam or ham.".lower()
    preprocessed_email = loaded_model.named_steps['tfidfvectorizer'].transform([new_email])
    prediction = loaded_model.predict(preprocessed_email)

    print("Prediction:", prediction)

if __name__ == '__main__':
    main()
