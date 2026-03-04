import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

model = pickle.load(open("spam_model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

email = "Congratulations! You won a free ticket. Call now!"

ps = PorterStemmer()
def preprocess(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words("english")]
    return " ".join(words)

processed = preprocess(email)
vector = cv.transform([processed]).toarray()
prediction = model.predict(vector)

print("spam" if prediction[0] == 1 else "Not spam")