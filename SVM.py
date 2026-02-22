from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

emails = [
    "Congratulations! You've won a free iPhone", "Claim your lottery prize now",
    "Exclusive deal just for you", "Act fast! Limited-time offer",
    "Hello, how are you today", "Please find the attached report",
    "Thank you for your support", "The project deadline is next week"
] 
labels = [1, 1, 1, 1, 0, 0, 0, 0] 

vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
X = vectorizer.fit_transform(emails)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

svm_model = LinearSVC(C=1.0)
svm_model.fit(X_train, y_train) 

new_email = [input("\nEnter a new email message: ")]
new_email_vectorized = vectorizer.transform(new_email)
prediction = svm_model.predict(new_email_vectorized)

if prediction[0] == 1:
    print("Result: The email is spam.")
else:
    print("Result: The email is not spam.") 
    