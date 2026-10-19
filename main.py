# TechNova AI Internship - Task 3
# Sentiment Analysis using Logistic Regression

import nltk
import random
import matplotlib.pyplot as plt

from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

# Download dataset if it is not available
nltk.download("movie_reviews", quiet=True)

print("=" * 55)
print("      TECHNOVA SENTIMENT ANALYSIS PROJECT")
print("=" * 55)

# Step 1: Load the movie review dataset
print("\nLoading movie reviews...")

documents = []

for category in movie_reviews.categories():
    for file_id in movie_reviews.fileids(category):
        review = movie_reviews.raw(file_id)
        documents.append((review, category))

# Shuffle the reviews
random.seed(42)
random.shuffle(documents)

texts = [review for review, label in documents]
labels = [label for review, label in documents]

print("Total reviews:", len(texts))
print("Positive reviews:", labels.count("pos"))
print("Negative reviews:", labels.count("neg"))

# Step 2: Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("\nTraining reviews:", len(X_train))
print("Testing reviews:", len(X_test))

# Step 3: Create the machine-learning pipeline
# TF-IDF converts text into numerical features.
# English stopwords are removed during vectorization.
# Logistic Regression classifies the sentiment.

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

# Step 4: Train the model
print("\nTraining the sentiment analysis model...")
model.fit(X_train, y_train)
print("Training completed successfully!")

# Step 5: Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 55)
print("                 MODEL RESULTS")
print("=" * 55)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Negative", "Positive"]
    )
)

# Step 6: Display and save the confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Negative", "Positive"],
    cmap="Blues"
)

plt.title("Sentiment Analysis - Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix saved as confusion_matrix.png")

# Step 7: Save the model accuracy graph
plt.figure()
plt.bar(
    ["Test Accuracy"],
    [accuracy * 100]
)
plt.ylim(0, 100)
plt.ylabel("Accuracy (%)")
plt.title("Sentiment Analysis Model Accuracy")
plt.tight_layout()
plt.savefig("accuracy_graph.png")
print("Accuracy graph saved as accuracy_graph.png")

plt.show()

# Step 8: Predict sentiment for user-entered reviews
print("\n" + "=" * 55)
print("          TEST YOUR OWN MOVIE REVIEW")
print("=" * 55)

print("Type 'exit' to stop.")

while True:
    user_review = input("\nEnter a movie review: ").strip()

    if user_review.lower() == "exit":
        print("Thank you for using Sentiment Analysis!")
        break

    if not user_review:
        print("Please enter a review.")
        continue

    result = model.predict([user_review])[0]
    probabilities = model.predict_proba([user_review])[0]
    confidence = max(probabilities) * 100

    if result == "pos":
        print("Predicted Sentiment: POSITIVE 😊")
    else:
        print("Predicted Sentiment: NEGATIVE 😞")

    print(f"Prediction confidence: {confidence:.2f}%")
    