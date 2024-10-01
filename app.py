import streamlit as st
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import PyPDF2  # To read PDF files

# Function to read text from PDF or TXT files
def read_file(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    else:  # Assuming it's a TXT file
        return file.read().decode("utf-8").strip()

# Function to highlight query terms in the document text
def highlight_query(text, query):
    for term in query.split():
        # Create a highlighted version of the term
        highlighted_term = f"<span style='background-color: yellow; color: black;'>{term}</span>"
        text = text.replace(term, highlighted_term)
    return text

# Vector Space Model
class VectorSpaceModel:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    def query(self, query_string):
        query_vector = self.vectorizer.transform([query_string])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)
        return similarities.flatten()

    def retrieve_top_n(self, query_string, n=5):
        scores = self.query(query_string)
        top_indices = np.argsort(scores)[-n:][::-1]
        
        # Ensure the document containing the query term is on top, if it exists
        if any(query_string in self.documents[i] for i in top_indices):
            top_documents = [(self.documents[i], scores[i]) for i in top_indices if query_string in self.documents[i]]
            if top_documents:
                top_documents = [(top_documents[0][0], scores[self.documents.index(top_documents[0][0])])] + top_documents[1:]
        
        else:
            top_documents = [(self.documents[i], scores[i]) for i in top_indices]
        
        return top_documents

# TF-IDF Model
class TFIDF:
    def __init__(self, documents):
        self.documents = documents
        self.doc_count = len(documents)
        self.term_freqs = [self._term_frequencies(doc) for doc in documents]
        self.idf = self._compute_idf()

    def _term_frequencies(self, doc):
        freq = defaultdict(int)
        for term in doc.split():
            freq[term] += 1
        return freq

    def _compute_idf(self):
        idf = defaultdict(int)
        for doc in self.documents:
            unique_terms = set(doc.split())
            for term in unique_terms:
                idf[term] += 1
        for term in idf:
            idf[term] = math.log((self.doc_count + 1) / (idf[term] + 1)) + 1  # Smoothing
        return idf

    def tfidf_score(self, doc_idx, query_terms):
        score = 0
        for term in query_terms:
            tf = self.term_freqs[doc_idx].get(term, 0)
            idf = self.idf.get(term, 0)
            score += tf * idf
        return score

    def rank(self, query):
        query_terms = query.split()
        scores = [(doc_idx, self.tfidf_score(doc_idx, query_terms)) for doc_idx in range(self.doc_count)]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Ensure document containing query term appears at the top, if present
        for term in query_terms:
            for i, (doc_idx, score) in enumerate(scores):
                if term in self.documents[doc_idx] and score > 0:
                    scores.insert(0, scores.pop(i))
                    break
        
        return scores

# BM25 Model
class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc.split()) for doc in documents]
        self.avg_doc_len = sum(self.doc_len) / len(documents)
        self.term_freqs = [self._term_frequencies(doc) for doc in documents]
        self.idf = self._compute_idf()

    def _term_frequencies(self, doc):
        freq = defaultdict(int)
        for term in doc.split():
            freq[term] += 1
        return freq

    def _compute_idf(self):
        idf = defaultdict(int)
        num_docs = len(self.documents)
        for doc in self.documents:
            unique_terms = set(doc.split())
            for term in unique_terms:
                idf[term] += 1
        for term in idf:
            idf[term] = math.log((num_docs - idf[term] + 0.5) / (idf[term] + 0.5) + 1)
        return idf

    def score(self, doc_idx, query_terms):
        score = 0
        doc_len = self.doc_len[doc_idx]
        for term in query_terms:
            if term in self.term_freqs[doc_idx]:
                tf = self.term_freqs[doc_idx][term]
                idf = self.idf.get(term, 0)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len)))
        return score

    def rank(self, query):
        query_terms = query.split()
        scores = [(doc_idx, self.score(doc_idx, query_terms)) for doc_idx in range(len(self.documents))]
        scores.sort(key=lambda x: x[1], reverse=True)

        # Ensure document containing query term appears at the top, if present
        for term in query_terms:
            for i, (doc_idx, score) in enumerate(scores):
                if term in self.documents[doc_idx] and score > 0:
                    scores.insert(0, scores.pop(i))
                    break
        
        return scores

# Streamlit App
st.title("DocuSearch")

# Create tabs
tab_vsm, tab_tfidf, tab_bm25 = st.tabs(["Vector Space Model", "TF-IDF", "BM25"])

# Function to handle file uploads and search
def handle_file_upload(tab_name):
    uploaded_files = st.file_uploader(f"Upload up to 10 documents for {tab_name}", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded_files:
        documents = [read_file(file) for file in uploaded_files]
        query = st.text_input(f"Enter your query for {tab_name}:")
        
        # Set max_value based on the number of uploaded documents
        max_docs = min(len(documents), 10)  # Limit to 10 as per file uploader
        top_n = st.number_input("Number of top documents to retrieve:", min_value=1, max_value=max_docs, value=min(5, max_docs))

        if st.button(f"Search with {tab_name}"):
            if tab_name == "Vector Space Model":
                model = VectorSpaceModel(documents)
                results = model.retrieve_top_n(query, n=top_n)
                st.write("Top documents:")
                for doc, score in results:
                    if score > 0:  # Display only if score is positive
                        highlighted_doc = highlight_query(doc, query)
                        st.markdown(f"Document: {highlighted_doc} | Score: {score:.4f}", unsafe_allow_html=True)

            elif tab_name == "TF-IDF":
                model = TFIDF(documents)
                rankings = model.rank(query)
                st.write("Ranked documents:")
                for doc_idx, score in rankings:
                    if score > 0:  # Display only if score is positive
                        highlighted_doc = highlight_query(documents[doc_idx], query)
                        st.markdown(f"Document {doc_idx}: {highlighted_doc} (Score: {score:.4f})", unsafe_allow_html=True)

            elif tab_name == "BM25":
                model = BM25(documents)
                rankings = model.rank(query)
                st.write("Ranked documents:")
                for doc_idx, score in rankings:
                    if score > 0:  # Display only if score is positive
                        highlighted_doc = highlight_query(documents[doc_idx], query)
                        st.markdown(f"Document {doc_idx}: {highlighted_doc} (Score: {score:.4f})", unsafe_allow_html=True)

# Handle each tab's file upload and search
with tab_vsm:
    handle_file_upload("Vector Space Model")

with tab_tfidf:
    handle_file_upload("TF-IDF")

with tab_bm25:
    handle_file_upload("BM25")
