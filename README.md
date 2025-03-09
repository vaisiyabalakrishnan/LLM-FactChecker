# LLM-FactChecker

This project, VeriFact, is an AI-powered misinformation detection and fact-checking web application designed to help users verify the accuracy of online content. Using Flask, the system processes user-inputted URLs or text, extracts key information, and evaluates the credibility of claims through natural language processing (NLP) and AI models.

The application features automated article summarization, named entity recognition (NER) for extracting key topics, and fact-checking using a LLaMA-based AI model. It queries search engines for related articles, compares the extracted claims against verified sources, assigns a truth score (Green = True, Red = False, Yellow = Unverified) and gives an evidence-based explanation with related articles for the verdict given. Additionally, it includes a user feedback mechanism, allowing users to rate fact-checking accuracy on a 1-10 scale. This feedback is stored in a structured JSON format and used for fine-tuning the AI model over time.

By combining advanced NLP techniques, machine learning, and real-time user feedback, this project aims to combat misinformation, promote digital literacy, and foster responsible information consumption in the digital age.
