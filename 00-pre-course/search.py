import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
resp = requests.get(url)
raw_docs = resp.json()

documents = []

for course in raw_docs:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

df = pd.DataFrame(documents, columns=['course', 'section', 'question', 'text'])
df.head()

docs_sample = [
    "Course starts on 15th Jan 2024",
    "Prerequisites listed on GitHub",
    "Submit homeworks after start date",
    "Registration not required for participation",
    "Setup Google Cloud and Python before course"
]


cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(docs_sample)

names = cv.get_feature_names_out()

df_docs = pd.DataFrame(X.toarray(), columns=names).T
df_docs

cv_tf = TfidfVectorizer(stop_words='english')
X = cv_tf.fit_transform(docs_sample)