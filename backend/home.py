import streamlit as st
from PIL import Image
from backend import summarize_article, extract_article_text, extract_entities, query_google, fact_check

st.session_state["summary"] = None
st.session_state["entities"] = None
st.session_state["related_articles"] = None
st.session_state["result"] = None

col1, col2, col3 = st.columns(3)

with col2:
    st.header("uhm actually")
    img = Image.open("/Users/lakshmi/Downloads/frontend/dog.jpeg")
    st.image(img)
    st.title("Fact-check your article")
    url = st.text_input("Enter URL")

with col1:
   st.write("Info about AI use.")
   if url:
        article_text = extract_article_text(url)
        if not article_text:
            print("Failed to extract article text")

        st.session_state["summary"] = summarize_article(article_text)
        if st.session_state["summary"]:
            st.write("Summary:")
            st.write(st.session_state["summary"])

        else:
            print("Failed to summarize article")
               

with col3:
    with st.spinner("Analysing your article...."):
        if st.session_state["summary"]:
            entities = extract_entities(st.session_state["summary"])
            st.session_state["entities"] = entities

            if not entities:
                print("Failed to extract entities")

            st.session_state["related_articles"] = query_google(entities)
            
            result = fact_check(st.session_state["summary"], st.session_state["related_articles"])
            st.session_state["result"] = result

            if result:
                st.write("Result:")
                st.write(f"{result["verdict"]}")
                st.write(f"Truth Score: {result["score"]}")
                st.write(f"Explanation: {result["explanation"]}")
            
                st.page_link("pages/idk.py", label="Explore related articles")
           
