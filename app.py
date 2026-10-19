from flask import Flask, render_template, request
import nltk
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# Prepare the dataset
nltk.download("movie_reviews", quiet=True)

documents = []

for category in movie_reviews.categories():
    for file_id in movie_reviews.fileids(category):
        documents.append(
            (movie_reviews.raw(file_id), category)
        )

texts = [review for review, label in documents]
labels = [label for review, label in documents]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# Create and train the model
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=20000,
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

print("Training sentiment model...")
model.fit(X_train, y_train)
print("Model training completed!")


@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    confidence = None
    review = ""

    if request.method == "POST":
        review = request.form.get("review", "").strip()

        if review:
            prediction = model.predict([review])[0]
            probabilities = model.predict_proba([review])[0]
            confidence = round(max(probabilities) * 100, 2)

            if prediction == "pos":
                sentiment = "Positive"
            else:
                sentiment = "Negative"

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence,
        review=review
    )


if __name__ == "__main__":
    app.run(debug=True)