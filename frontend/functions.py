from transformers import pipeline
import spacy
import requests
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient
import json


# Load HuggingFace models for summarization and topic classification
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Load spaCy model for NER (named entity recognition)
nlp = spacy.load("en_core_web_sm")


# Extract text from article (user inputs url into our website)
def extract_article_text(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        # Raise an error for bad status codes
        response.raise_for_status()
        # Parse the raw HTML content into a structured format
        soup = BeautifulSoup(response.text, "html.parser")

        # Try different HTML tags
        article_text = ""

        # Extract from <article> tag
        article_tag = soup.find("article")
        if article_tag:
            article_text = " ".join([p.get_text() for p in article_tag.find_all("p")])

        # If <article> tag is empty, try <div> with article-like classes
        if not article_text.strip():
            for div in soup.find_all("div"):
                if "article" in div.get("class", []) or "content" in div.get("class", []):
                    article_text = " ".join([p.get_text() for p in div.find_all("p")])
                    if article_text:
                        break  # Stop if we found content

        # If still empty, try generic <p> and <span> tags
        if not article_text.strip():
            paragraphs = soup.find_all("p")
            spans = soup.find_all("span")
            article_text = " ".join([p.get_text() for p in paragraphs]) + " ".join([s.get_text() for s in spans])
        
        return article_text

    except Exception as e:
        print(f"Error extracting article text: {e}")
        return None


# Summarise article (HuggingFace)
def summarize_article(text: str) -> str:
    try:
        summary = summarizer(text, max_length=100, min_length=20, do_sample=False)
        return summary[0]["summary_text"]

    except Exception as e:
        print(f"Error summarizing article text: {e}")
        return None


# Extract entities from summary (spaCy)
def extract_entities(summary: str) -> list:
    try:
        doc = nlp(summary)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities[:5]
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return []


# Perform a search using SerpAPI
def search(query: str) -> list:
    params = {
        "q": query,
        "api_key": "3c31e8239b027eddbb900a8c072758ab8635ab3daf17326a2444ab90857b618a" 
    }
    response = requests.get("https://serpapi.com/search", params=params)
    if response.status_code == 200:
        return response.json().get("organic_results", [])
    else:
        print(f"Error searching: {response.status_code}")
        return []

# Query public google for relevant evidence
def query_google(entities: list) -> list:
    try:
        # Combine entities into a search query
        query_terms = [entity[0] for entity in entities]
        query = " ".join(query_terms)
        print(f"Search query: {query}")

        search_results = search(query)
        
        related_articles = []

        for result in search_results:
            title = result.get("title", "")
            snippet = result.get("snippet", "").replace("\n", " ")
            link = result.get("link", "")
            related_articles.append({
                "Title": title,
                "Snippet": snippet,
                "Link": link
            })
        
        return related_articles

    except Exception as e:
        print(f"Error querying databases: {e}")
        return []
    
def fact_check(summary, related_articles):
    client = InferenceClient(
	provider="novita",
	api_key="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")
    
    messages = [
        {"role": "system", "content": 
            "You are a highly accurate fact-checking AI. "
            "You verify articles by referring to the related articles provided. If not, refer to other reliable sources."
            "Classify articles as TRUE, FALSE, or UNVERIFIED."
            "Provide a truth score from 0 to 100 based on evidence. For unverified sources, assign a score based on the likelihood of the claim's truth."
            "Provide a short explanation based on evidence. Quote the articles in your response."
        },
        {"role": "user", "content": 
            f"Summary: {summary}\n Related articles:\n{related_articles}\n\n"
            "Evaluate the claim and return the result in the following format as JSON:\n"
            '{"verdict": "TRUE/FALSE/UNVERIFIED", "score": XX, "explanation": "short explanation based on the evidence"}'
        }
    ]

    completion = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct", 
	messages=messages, 
	max_tokens=500,)

    response = completion.choices[0].message.content
    response = response.strip()
    
    try:
        result = json.loads(response)
    
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return {"error": "Invalid JSON format", "response": response}

    return result


