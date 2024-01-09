import pandas as pd
import re

from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv(r'C:\Users\Carina\Desktop\Master\An 2\Proiectarea sistemelor de inteligenta artificiala\Sentiment '
                 r'Analysis Project\DisneylandReviews.csv', encoding='ISO-8859-1')
df = df[['Rating', 'Review_Text']]
print(df.head())


def create_sentiment(rating):
    if rating in [1, 2]:
        return -1
    elif rating in [4, 5]:
        return 1
    else:
        return 0


df['Sentiment'] = df['Rating'].apply(create_sentiment)
print("\nDataset with new column Sentiment:")
print(df)


def clean_and_tokenize_data(review):
    no_punc = re.sub(r'[^\w\s]', '', review)
    no_digits = ''.join([i for i in no_punc if not i.isdigit()])
    stop_words = set(stopwords.words('english'))
    preprocessed_text = ' '.join([word.lower() for word in word_tokenize(no_digits) if word.lower() not in stop_words])
    return preprocessed_text


df['Preprocessed Review_Text'] = df['Review_Text'].apply(clean_and_tokenize_data)
print("\nDataset with preprocessed reviews:")
print(df[['Review_Text', 'Preprocessed Review_Text']].head())

df = df.drop('Review_Text', axis=1)
print("\nDataset:")
print(df)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Preprocessed Review_Text'])
y = df['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regression Model
logistic_regression_model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
logistic_regression_model.fit(X_train, y_train)
pred_lr = logistic_regression_model.predict(X_test)

print(f'Logistic Regression Model Accuracy: {accuracy_score(pred_lr, y_test)}')
precision = precision_score(y_test, pred_lr, average='weighted')
print(f'Logistic Regression Model Precision: {precision}')
recall = recall_score(y_test, pred_lr, average='weighted')
print(f'Logistic Regression Model Recall: {recall}')
f1 = f1_score(y_test, pred_lr, average='weighted')
print(f'Logistic Regressio Model F1 Score: {f1}')

# SVM Model
svm_model = svm.SVC(kernel='linear', cache_size=2000)
svm_model.fit(X_train, y_train)
pred_svm = svm_model.predict(X_test)

print(f'SVM Model Accuracy: {accuracy_score(pred_svm, y_test)}')
precision = precision_score(y_test, pred_svm, average='weighted')
print(f'SVM Model Precision: {precision}')
recall = recall_score(y_test, pred_svm, average='weighted')
print(f'SVM Model Recall: {recall}')
f1 = f1_score(y_test, pred_svm, average='weighted')
print(f'SVM Model F1 Score: {f1}')
