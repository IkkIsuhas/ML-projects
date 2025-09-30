import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_word = set(stopwords.words('english'))

dataset = pd.read_csv('IMDB.csv')
# dataset.head()

# dataset.isnull().sum()

def preprocess(text):
  try:
    if not isinstance(text, str):
      return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]","",text)
    words = text.split()
    words = [word for word in words if word not in stop_word]
    return" ".join(words)

  except Exception as e:
    print(f"Error in preprocessing{e}")
    return ""

try:
  dataset['clean_review'] = dataset['review'].apply(preprocess)
  dataset.to_csv('IMDB2.csv',index=False)
  df = pd.read_csv('IMDB2.csv')
  df['sentiment'] = df['sentiment'].map({'positive':1,'negative':0})

  vector = TfidfVectorizer()
  x = vector.fit_transform(df['clean_review'])
  y = df['sentiment']
  x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
  model = MultinomialNB()
  model.fit(x_train,y_train)
  # pred = model.predict(x_test)

except FileNotFoundError:
  print("Error: IMDB.csv file not found. Please check the path.")
except Exception as e:
  print(f"Unexpected error occurred while loading/training: {e}")

def predict_statement(review_text):
  try:
    cleaned = preprocess(review_text)
    if cleaned.strip() == "":
      return "Invalid input. Please enter a valid statement."
    features = vector.transform([cleaned])
    prediction = model.predict(features)[0]
    sentement_map = {1:'positive:)',0:'negative:('}
    return sentement_map[prediction]
  except Exception as e:
    print(f"Prediction Error: {e}")

try:
  statemnt = input("Enter a statement: ")
  print(predict_statement(statemnt))
except KeyboardInterrupt:
  print("Program interrupted by the user.")
except Exception as e:
  print(f"Unexpected error occurred: {e}")