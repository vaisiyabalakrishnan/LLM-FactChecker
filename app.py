from flask import Flask, render_template, request, session
from flask_session import Session
from functions import summarize_article, extract_article_text, extract_entities, query_google, fact_check, save_feedback


app = Flask(__name__)

app.config['SESSION_TYPE'] = 'filesystem'
Session(app)


@app.route("/", methods=["GET", "POST"])
def home():
    form_type = request.form.get("type", request.args.get("type", "url"))
    return render_template("home.html", error=None, form_type=form_type)

@app.route("/results", methods=["GET", "POST"])
def results():
    message = None
    summary = None
    related_articles = None
    result = None

    if request.method == "POST":
        if "feedback" in request.form:  
            # Handle feedback submission
            rating = request.form.get("feedback")
            message = "Thank you for your feedback!"
            
            # Save feedback with retrieved data
            save_feedback(session["summary"], session["result"], int(rating))
        
        else: 
            url = request.form.get("url", None)
            text = request.form.get("text", None)
                
            if url:
                try:
                    article_text = extract_article_text(url)
                    if not article_text:
                        return render_template("home.html", error="Invalid URL. Could not extract text.")
            
                    try:
                        summary = summarize_article(article_text)
                        session["summary"] = summary
                    except Exception as e:
                        return render_template("home.html", error=str(e))
                
                except Exception as e:
                    return render_template("home.html", error="Invalid URL. Please enter a valid URL.")

            elif text:
                summary = summarize_article(text)
                session["summary"] = summary
                
            try:
                entities = extract_entities(summary)
                related_articles = query_google(entities)
                result = fact_check(text, related_articles)
                session["result"] = result

            except Exception as e:
                return render_template("home.html", error=str(e))
    
    return render_template("results.html", summary=summary, related_articles=related_articles, result=result, message = message)
    

