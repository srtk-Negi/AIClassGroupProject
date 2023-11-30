import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.base import TransformerMixin
import joblib
from nltk.corpus import stopwords
import nltk

class LowercaseTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [text.lower() for text in X]

def create_train_model(X_train, y_train):
    model = make_pipeline(
        LowercaseTransformer(),
        TfidfVectorizer(stop_words=stopwords.words('english'),
                        strip_accents='ascii',
                        token_pattern=r'\b\w\w+\b'),
        SVC(kernel='linear')
    )
    model.fit(X_train, y_train)

    return model

def save_model(model):
    joblib.dump(model, 'spam_classifier_model.joblib')


def load_model():
    return joblib.load('spam_classifier_model.joblib')


def make_predictions_on_new_email(model, new_email):
    prediction = model.predict([new_email])
    print("Prediction:", prediction[0])


def make_predictions_on_test_data(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

def main():
    df = pd.read_csv('spam_ham_dataset.csv')
    # print(df.head())

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    nltk.download('stopwords')

    model = create_train_model(X_train, y_train)

    make_predictions_on_test_data(model, X_test, y_test)

    save_model(model)

    loaded_model = load_model()


    new_ham_email = "Subject: Card Submission; Hello David, Hope this email finds you well. This email is just to remind you that we still haven't recived your application. It is due in 2 days and we recommend that you submit the application ASAP on the portal or in person at our office. Thank You"

    new_spam_email = "Subject: You've won a free ticket; Congratulations! You've won a free ticket to the USA this summer. Click here to claim your prize."

    make_predictions_on_new_email(loaded_model, new_spam_email)

if __name__ == '__main__':
    main()
