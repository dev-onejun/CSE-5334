"""
CSE-5334-DATA MINING
Professor: Dr. Marnim Galib

Writer: Wonjun Park
UTA ID: 1002237177
wxp7177@mavs.uta.edu

Programming Assignment 1
Fall 2024, Computer Science and Engineering, University of Texas at Arlington

- Lint with Black
"""

import os
from math import log

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

CORPUS_ROOT = "./US_Inaugural_Addresses"
N = 40


def read_files(CORPUS_ROOT: str) -> list[str]:
    docs = []

    for filename in os.listdir(CORPUS_ROOT):
        if (
            filename.startswith("0")
            or filename.startswith("1")
            or filename.startswith("2")
            or filename.startswith("3")
            or filename.startswith("4")
        ):
            file = open(
                os.path.join(CORPUS_ROOT, filename), "r", encoding="windows-1252"
            )
            doc = file.read()
            file.close()
            doc = doc.lower()
            docs.append(doc)

    N = len(docs)

    return docs


def tokenize(docs: list[str]) -> (list[list[str]], list[str]):
    tokens = set()
    tokenized_docs = []

    tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
    for doc in docs:
        tokens_per_doc = tokenizer.tokenize(doc)

        for token in tokens_per_doc:
            tokens.add(token)
        tokenized_docs.append(tokens_per_doc)

    return tokenized_docs, list(tokens)


def preprocess(tokenized_docs: list[list[str]], tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    STOPWORDS = stopwords.words("english")

    preprocessed_tokens = [
        stemmer.stem(token) for token in tokens if token not in STOPWORDS
    ]

    return preprocessed_tokens


docs = read_files(CORPUS_ROOT)
tokenized_docs, tokens_set = tokenize(docs)
preprocessed_docs, corpus = preprocess(tokenized_docs, tokens_set)


def getidf(token) -> float:
    df_t = 0
    for doc in tokenized_docs:
        if token in doc:
            df_t += 1

    return log(N / df_t, 10)


def compute_TF_IDFs() -> list[dict[str, float]]:
    TF_IDFs = []
    for doc in tokenized_docs:
        TF_IDF = {}
        for token in doc:
            if token in TF_IDF:
                TF_IDF[token] += 1
            else:
                TF_IDF[token] = 1

        for token in TF_IDF:
            TF_IDF[token] = TF_IDF[token] / len(doc) * getidf(token)

        TF_IDFs.append(TF_IDF)

    return TF_IDFs


# def getweight(filename, token) -> :
# Return the normalized TF-IDF weight of a token in the document named 'filename'.
# If the token does not exist in the document, return 0.
# Note that steming the parameter 'token' is necessary before calculating the score.


def query(qstring):
    pass


if __name__ == "__main__":
    print("%.12f" % getidf("democracy"))
    print("%.12f" % getidf("foreign"))
    print("%.12f" % getidf("states"))
    print("%.12f" % getidf("honor"))
    print("%.12f" % getidf("great"))

    # getidf(token)
    # getweight(token)
    # query(query_strings)
