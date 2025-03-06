from fastapi import FastAPI, HTTPException
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup

# Initialise FastAPI app
app = FastAPI()

# Load spaCy model for NER (named entitiy recognition)
nlp = spacy.load("en_core_web_sm")

# Load HuggingFace models for summarization and topic classification
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load embedding model (to convert text into numerical vectors)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Google Fact Check API
api_key = "AIzaSyAsTvUkuOZBLSAOu5WeekJMdyoO7siQ-oE"
fact_check_service = build("factchecktools", "v1alpha1", developerKey=api_key)


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


# Query Google Fact Check API for fact checks
def query_fact_check(entities: list, topic: str) -> list:
    # Combine entities and topic into a search query
    query_terms = [entity[0] for entity in entities] + [topic]
    query = " ".join(query_terms)

    try:
        request = fact_check_service.claims().search(query=query)
        response = request.execute()

        fact_checks = []
        if 'claims' in response:
            for claim in response['claims']:
                for review in claim.get("claimReview", []):
                    fact_checks.append({
                        "title": review.get("title", "No title"),
                        "summary": review.get("text", "No summary available"),
                        "source": review.get("publisher", {}).get("name", "Unknown source"),
                        "url": review.get("url", "#"),
                        "rating": review.get("textualRating", "No rating")
                    })
        return fact_checks
    
    except Exception as e:
        print(f"Error querying Google Fact Check API: {e}")
        return []


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

        ## Query Google Fact Check API for fact checks
        evidence = query_fact_check(entities, topic)
        print(f"Evidence: {evidence}")

        # Format the results
        formatted_results = []
        for doc in evidence:
            formatted_results.append({
                "title": doc["title"],
                "summary": doc["summary"],
                "source": doc["source"],
                "url": doc["url"],
                "rating": doc["rating"]
            })

        return {
            "summary": summary,
            "entities": entities,
            "topic": topic,
            "documents": formatted_results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# API endpoint
@app.post("/process-url")
async def process_url_endpoint(url: str):
    return process_url(url)

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
