from flask import Flask,request,render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from nltk.corpus import stopwords


stemmer=PorterStemmer()
tfidf=TfidfVectorizer()
random=RandomForestClassifier()
app=Flask(__name__)
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from nltk.corpus import stopwords
import pickle

nltk.download('stopwords')

# Load pre-trained models (replace with your paths)
# tfidf = pickle.load(open('tfidf.pkl', 'rb'))
# random = pickle.load(open('model.pkl', 'rb'))

stemmer = PorterStemmer()
app = Flask(__name__)

def predict_text(text):
    data = re.sub('[^a-zA-Z]', ' ', text)
    data = data.lower()
    words = [stemmer.stem(word) for word in data.split() if word not in stopwords.words('english')]
    cleaned_text = ' '.join(words)
    
    # Fit on single input - not ideal but works for testing
    tfidf.fit([cleaned_text])  
    vectorized = tfidf.transform([cleaned_text])
    # random.fit(vectorized)
    prediction = random.predict(vectorized)
    return prediction


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        text = request.form.get('text')
        if not text:
            return "No text provided!", 400
        prediction = predict_text(text)
        return render_template('home.html', prediction=prediction)
    # else:
    #     # If user visits /predict via browser, redirect to form
    #     return render_template('index.html')

    

if __name__ == "__main__":
    app.run(debug=True)
