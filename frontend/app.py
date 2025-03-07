from flask import Flask, render_template, request
from functions import summarize_article, extract_article_text, extract_entities, query_google, fact_check

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        url = request.form.get("url")
        if not url:
            return render_template("home.html", error = "Please enter a URL")
        
        try:
            article_text = extract_article_text(url)
        except Exception as e:
            error = "Error extracting article text"
            return render_template("home.html", error = error)
            
        if article_text:
            try:
                summary = summarize_article(article_text)
            except Exception as e:
                error = "Error summarizing article text"
                return render_template("home.html", error = error)
            
            try:
                entities = extract_entities(summary)
            except Exception as e:
                error = "Error extracting entities"
                return render_template("home.html", error = error)
            
            try: 
                related_articles = query_google(entities)
            except Exception as e:
                error = "Error querying databases"
                return render_template("home.html", error = error)
            
            try:
                result = fact_check(summary, related_articles)
            except Exception as e:
                error = "Error fact-checking"
                return render_template("home.html", error = error)
            
            try:
                return render_template("results.html", summary=summary, related_articles = related_articles, result = result)
        
            except Exception as e:
                return render_template("home.html", error = "Could not render result.html")
            
        else:
            return render_template("home.html", error = "Could not extract article text")
    
    return render_template("home.html", error = None)