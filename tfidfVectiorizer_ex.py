from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "I love machine learning",
    "I love programming",
    "Machine learning is fun",
    "Programming is fun"
]

vect = TfidfVectorizer()
X = vect.fit_transform(documents)
X_ts = vect.transform(documents)
print(X)
print('tans',X_ts)
print(vect.get_feature_names_out())