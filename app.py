import streamlit as st
import json
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

# Silent NLTK check
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet',quiet=True)
    
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt',quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords',quiet=True)

# Load + process data only once (IMPORTANT)
@st.cache_resource
def load_data():
    with open(datathirdd.json, "r", encoding="utf-8") as file:
        data = json.load(file)

    faqs = data["faqs"]
    questions = [item["question"] for item in faqs]
    answers = [item["answer"] for item in faqs]

    return questions, answers

questions, answers = load_data()



# Preprocessing
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return " ".join(words)

clean_questions = [preprocess(q) for q in questions]

#Cache vectorizer too (VERY IMPORTANT)

@st.cache_resource
def train_model(questions):
    clean_questions = [preprocess(q) for q in questions]
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    X = vectorizer.fit_transform(clean_questions)
    return vectorizer, X

vectorizer, X = train_model(questions)

# Chatbot function
def chatbot_response(user_input):
    user_input_clean = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_clean])
    
    similarity = cosine_similarity(user_vector, X)
    score = similarity.max()
    index = similarity.argmax()
    
    print("DEBUG SCORE:", score) 
    
    if score < 0.35:
        return "Sorry, I didn't understand that. Please ask more clearly."
    
    return answers[index]


# UI
st.title("🎓 ABC Institute Chatbot")
st.write("Ask me anything about the college!")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.text_input("You:")

if user_input:
    response = chatbot_response(user_input)
    st.session_state.chat.append(("You", user_input))
    st.session_state.chat.append(("Bot", response))

for sender, message in st.session_state.chat:
    st.write(f"**{sender}:** {message}")
