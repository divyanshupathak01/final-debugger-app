# app.py - Final TF-IDF Version

import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Debugging Helper",
    page_icon="🤖",
    layout="wide"
)

# --- Preprocessing Function ---
# This needs to be available for processing the user's query
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return ' '.join(tokens)

# --- Model and Data Loading ---
@st.cache_resource
def load_assets():
    """Loads all necessary models and dataframes."""
    print("Loading assets...")
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('tfidf_matrix.pkl', 'rb') as f:
        matrix = pickle.load(f)
    
    questions_df = pd.read_csv('processed_questions.csv')
    answers_df = pd.read_csv('processed_answers.csv')
    
    print("Assets loaded.")
    return vectorizer, matrix, questions_df, answers_df

vectorizer, matrix, df_questions, df_answers = load_assets()

# --- Core Functions ---
def find_similar_questions(query, top_n=10):
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])
    cosine_similarities = cosine_similarity(query_vector, matrix).flatten()
    related_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    
    results = df_questions.iloc[related_indices].copy()
    results['SimilarityScore'] = cosine_similarities[related_indices]
    return results

def get_suggested_solution(question_id):
    try:
        accepted_answer_id = df_questions.loc[df_questions['QuestionId'] == question_id, 'AcceptedAnswerId'].iloc[0]
        answer_body = df_answers.loc[df_answers['AnswerId'] == accepted_answer_id, 'AnswerBody'].iloc[0]
        soup = BeautifulSoup(answer_body, 'html.parser')
        code_blocks = [code.get_text() for code in soup.find_all('code')]
        return code_blocks if code_blocks else ["No code blocks found."]
    except (IndexError, KeyError):
        return ["Could not find a corresponding answer."]

# --- User Interface ---
st.title("🤖 AI-Powered Python Debugging Helper")
st.markdown("This tool uses a TF-IDF machine learning model to find the most relevant Stack Overflow posts for your Python errors.")
st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    user_code = st.text_area("Paste your Python code here:", height=300, placeholder="import pandas as pd\n...")
with col2:
    user_error = st.text_area("Paste the full error message here:", height=300, placeholder="KeyError: 'column_name'")

if st.button("Find Solution", type="primary", use_container_width=True):
    if user_code and user_error:
        with st.spinner("🧠 Analyzing your query..."):
            user_query = user_code + "\n" + user_error
            top_matches = find_similar_questions(user_query)
            
            st.markdown("---")
            st.subheader("🏆 Top Results")
            
            top_question = top_matches.iloc[0]
            st.success(f"**Best Match:** [{top_question['Title']}](https://stackoverflow.com/q/{top_question['QuestionId']}) (Similarity: {top_question['SimilarityScore']:.2f})")
            
            with st.expander("💡 **View Suggested Solution Code**", expanded=True):
                solution_code = get_suggested_solution(top_question['QuestionId'])
                if solution_code and "No code blocks found" not in solution_code[0]:
                    for i, snippet in enumerate(solution_code):
                        st.code(snippet, language='python', line_numbers=True)
                else:
                    st.warning("No code snippets were found in the top-ranked answer.")
            
            with st.expander("➕ **View Other Similar Questions**"):
                for i, row in top_matches.iloc[1:].iterrows():
                    st.markdown(f"* [{row['Title']}](https://stackoverflow.com/q/{row['QuestionId']}) (Similarity: {row['SimilarityScore']:.2f})")
    else:
        st.error("Please enter both your code and the error message to get a solution.")