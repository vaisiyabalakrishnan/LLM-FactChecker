from fastapi import FastAPI, HTTPException
import spacy
from transformers import pipeline
import pinecone
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch

# Initialise FastAPI app
app = FastAPI()

# Load spaCy model for NER (named entitiy recognition)
nlp = spacy.load("en_core_web_sm")

# Load HuggingFace models for summarization and topic classification
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Initialize Pinecone (vector database)
pinecone.init(api_key="pcsk_5JwvRy_BxzmowFyrYy8HKEdeXZosURrcsfhEUM2QcPssEjE1A4nRUSFQ7MVESuDFqNGVJp")
index = pinecone.Index("fact-checking-db")

# Load embedding model (to convert text into numerical vectors)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# Extract text from article (user inputs url into our website)
def extract_article_text(url: str) -> str:
    try:
        response = requests.get(url)
        # Raise an error for bad status codes
        response.raise_for_status()
        # Parse the raw HTML content into a structured format
        soup = BeautifulSoup(response.text, "html parser")

        # Extract all paragraph texts
        paragraphs = soup.find_all("p")
        article_text = " ".join([p.get_text() for p in paragraphs])
        return article_text

    except Exception as e:
        print(f"Error extracting article text: {e}")
        return None


# Summarise article (HuggingFace)
def summarize_article(text: str) -> str:
    try:
        summary = summarizer(text, max_length=130, min_length=15, do_sample=False)
        return summary[0]["summary_text"]

    except Exception as e:
        print(f"Error summarizing article text: {e}")
        return None


# Extract entities from summary (spaCy)
def extract_entities(summary: str) -> list:
    doc = nlp(summary)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities


# Classify topic (HuggingFace)
def classify_topic(text: str) -> str:
    candidate_labels = ["science", "health", "politics", "technology", "sports"]
    result = classifier(text, candidate_labels)
    # Return most likely topic
    return result["labels"][0] 

# Query public fact-checking databases for relevant evidence
def query_databases(entities: list, topic: str) -> list:
    # Combine entities and topic into a search query
    query_terms = []
    for entity in entities:
        query_terms.append(entity[0])
    query_terms.append(topic)
    query = " ".join(query_terms)

    # Perform a search (SerpAPI)
    def search(query):
        params = {
            "q": query,
            "api_key": "a17f44b09203fd9e3955a53dfc6c9acc9dba5fa5"
        }
        response = requests.get("https://serpapi.com/search", params=params)
        if response.status_code == 200:
            return response.json().get("organic_results", [])
        else:
            print(f"Error searching: {response.status_code}")
            return []
    
    search_results = search(query)
    return search_results


# Retrieve semantically similar documents to summary
def retrieve_documents(summary: str) -> list:
    # Convert summary into vector (SentenceTransformer)
    summary_vector = embedding_model.encode(summary).tolist()

    # Query Pinecone for similar documents
    results = index.query(summary_vector, top_k=5, include_metadata=True)
    return [match["metadata"] for match in results["matches"]]

# Main function
def process_url(url: str) -> dict:
    try:
        # Extract article text
        article_text = extract_article_text(url)
        if not article_text:
            raise HTTPException(status_code=400, detail="Failed to extract article text")

        # Summarize article
        summary = summarize_article(article_text)
        if not summary:
            raise HTTPException(status_code=400, detail="Failed to summarize article")

        print(f"Summary: {summary}")

        # Extract entities from the summary
        entities = extract_entities(summary)
        print(f"Entities: {entities}")

        # Classify topic
        topic = classify_topic(summary)
        print(f"Topic: {topic}")

        # Query public databases for evidence
        evidence = query_databases(entities, topic)
        print(f"Evidence: {evidence}")

        # Retrieve similar documents from Pinecone
        similar_docs = retrieve_documents(summary)
        print(f"Similar Documents: {similar_docs}")

        # Combine results
        all_documents = evidence + similar_docs

        # Format the results
        formatted_results = []
        for doc in all_documents:
            formatted_results.append({
                "title": doc.get("title", "No title"),
                "summary": doc.get("snippet", "No summary available")
            })

        return {
            "summary": summary,
            "entities": entities,
            "topic": topic,
            "documents": formatted_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
