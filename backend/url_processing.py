from fastapi import FastAPI, HTTPException
from transformers import pipeline
import spacy
import requests
from bs4 import BeautifulSoup

# Initialise FastAPI app
app = FastAPI()

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
        summary = summarizer(text, max_length=80, min_length=15, do_sample=False)
        return summary[0]["summary_text"]

    except Exception as e:
        print(f"Error summarizing article text: {e}")
        return None


# Extract entities from summary (spaCy)
def extract_entities(summary: str) -> list:
    try:
        doc = nlp(summary)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities[:4]
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
        
        evidence = []

        for result in search_results:
            title = result.get("title", "")
            snippet = result.get("snippet", "").replace("\n", " ")
            link = result.get("link", "")
            evidence.append({
                "Title": title,
                "Snippet": snippet,
                "Link": link
            })
        
        return evidence

    except Exception as e:
        print(f"Error querying databases: {e}")
        return []


# # Query Google Fact Check API for fact checks
# def fact_check_claim(summary: str) -> list:
#     try:
#         url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={summary}&languageCode=en&key={api_key}"
        
#         data = requests.get(url).json()
       
#         results = []

#         if "claims" in data:
#             for claim in data["claims"]:
#                 claim_text = claim.get("text", "No claim text found")
#                 for review in claim.get("claimReview", []):
#                     results.append({
#                         "claim": claim_text,
#                         "fact_checker": review["publisher"]["name"],
#                         "rating": review["textualRating"],
#                         "url": review["url"]
#                     })
        
#         return results
    
#     except Exception as e:
#         print(f"Error querying Google Fact Check API: {e}")
#         return []


# Main function
def main(url: str) -> dict:
    try:
        article_text = extract_article_text(url)
        if not article_text:
            raise HTTPException(status_code=400, detail="Failed to extract article text")

        summary = summarize_article(article_text)
        if not summary:
            raise HTTPException(status_code=400, detail="Failed to summarize article")

        entities = extract_entities(summary)
        if not entities:
            raise HTTPException(status_code=400, detail="Failed to extract entities")

        evidence = query_google(entities)

        return {
            "Summary": summary,
            "Entities": entities,
            "Evidence": evidence
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# API endpoint
@app.post("/process-url")
async def main_endpoint(url: str):
    return main(url)

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
