import spacy
from textblob import TextBlob

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")

# Sample Amazon reviews
reviews = [
    "I love my new iPhone 13! Apple really nailed it.",
    "The Samsung Galaxy S21 is overpriced and underwhelming.",
    "Amazon's Echo Dot is surprisingly good for the price.",
    "Terrible experience with the Lenovo laptop. Never buying again."
]

# Process each review
for review in reviews:
    doc = nlp(review)
    print(f"\nReview: {review}")
    
    # Named Entity Recognition
    print("Entities:")
    for ent in doc.ents:
        if ent.label_ in ["PRODUCT", "ORG"]:
            print(f"  - {ent.text} ({ent.label_})")
    
    # Rule-based Sentiment Analysis
    sentiment = TextBlob(review).sentiment.polarity
    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    print(f"Sentiment: {sentiment_label} (Score: {sentiment:.2f})")