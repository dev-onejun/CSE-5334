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
import math

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

CORPUS_ROOT = "./US_Inaugural_Addresses"


def read_files(CORPUS_ROOT: str) -> dict[str, str]:
    docs = {}

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
            docs.update({filename: doc})

    return docs


def tokenize(docs: dict[str, str]) -> (dict[str, list[str]], list[str]):
    """
    tokenize() -> (tokenized_docs, tokens)
    * tokenized_docs: all the tokens in each document
    * tokens: all the tokens in the collection
    """
    tokens = set()
    tokenized_docs = {}

    tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
    for filename, doc in docs.items():
        tokens_per_doc = tokenizer.tokenize(doc)

        for token in tokens_per_doc:
            tokens.add(token)
        tokenized_docs.update({filename: tokens_per_doc})

    return tokenized_docs, list(tokens)


def preprocess(
    tokenized_docs: dict[str, list[str]], tokens: list[str]
) -> (dict[str, list[str]], list[str]):
    """
    preprocess() -> (preprocessed_docs, corpus)
    * preprocessed_docs: all the preprocessed tokens in each document
        - dict{filename: list of preprocessed tokens}
    * corpus: all the preprocessed tokens in the collection
        - list[preprocessed tokens]
    """
    stemmer = PorterStemmer()
    STOPWORDS = stopwords.words("english")

    preprocessed_docs = {}
    for filename, doc in tokenized_docs.items():
        preprocessed_docs[filename] = [
            stemmer.stem(token) for token in doc if token not in STOPWORDS
        ]

    preprocessed_tokens = [
        stemmer.stem(token) for token in tokens if token not in STOPWORDS
    ]

    return preprocessed_docs, preprocessed_tokens


def compute_idf(N: int, preprocessed_docs: dict[str, list[str]], token: str) -> float:
    df_t = 0
    for doc in preprocessed_docs.values():
        if token in doc:
            df_t += 1
            continue

    try:
        return log(N / df_t, 10)
    except ZeroDivisionError:
        return -1.0


def compute_TF_IDFs(
    preprocessed_docs: dict[str, list[str]],
    corpus: list[str],
) -> (dict[[str, dict[str, float]]], dict[str, float]):
    """
    compute_TF_IDFs() -> (TF_IDFs, IDF_vectors)
    * TF_IDFs: TF-IDF vectors for each document
        - dict{filename: dict{token: TF-IDF}}
    * IDF_vectors: IDF vectors for each token
        - dict{token: IDF}
    """
    N = len(preprocessed_docs)

    TF_IDFs, IDF_vectors = {}, {}
    for filename, doc in preprocessed_docs.items():
        TF_IDF = {}
        for token in doc:
            if token in TF_IDF:
                TF_IDF[token] += 1
            else:
                TF_IDF[token] = 1

        for token in TF_IDF:
            idf = compute_idf(N, preprocessed_docs, token)

            TF_IDF[token] = (1 + log(TF_IDF[token], 10)) * idf
            IDF_vectors[token] = idf

        try:
            norm = math.sqrt(sum(value**2 for value in TF_IDF.values()))
            for token in TF_IDF:
                TF_IDF[token] /= norm
        except ZeroDivisionError:
            print(
                "ZeroDivisionError occurred while normalizing TF-IDF vectors. Check the data."
            )
            pass

        TF_IDFs[filename] = TF_IDF
        # TF_IDFs.update({filename: TF_IDF})

    return TF_IDFs, IDF_vectors


def getidf(token) -> float:
    stemmer = PorterStemmer()
    token = stemmer.stem(token)

    return IDF_vectors[token]


def getweight(filename, token) -> float:
    stemmer = PorterStemmer()
    token = stemmer.stem(token)

    try:
        return TF_IDF_vectors[filename][token]
    except KeyError:
        return 0


def get_query_vector(qstring) -> dict[str, float]:
    idf_for_query = 1  # ltc.lnc

    qstring = qstring.lower()
    qstring = {"query": qstring}
    tokenized_query, _ = tokenize(qstring)
    preprocessed_query, _ = preprocess(tokenized_query, _)

    query_vector = {}
    for token in preprocessed_query["query"]:
        if token in query_vector:
            query_vector[token] += 1
        else:
            query_vector[token] = 1

    for token in query_vector:
        query_vector[token] = (1 + log(query_vector[token], 10)) * idf_for_query

    try:
        norm = math.sqrt(sum(value**2 for value in query_vector.values()))
        for token in query_vector:
            query_vector[token] /= norm
    except ZeroDivisionError:
        print(
            "ZeroDivisionError occurred while normalizing query vector. Check the data."
        )
        pass

    return query_vector


def query(qstring):
    query_vector = get_query_vector(qstring)


docs = read_files(CORPUS_ROOT)
tokenized_docs, tokens_set = tokenize(docs)
preprocessed_docs, corpus = preprocess(tokenized_docs, tokens_set)

TF_IDF_vectors, IDF_vectors = compute_TF_IDFs(preprocessed_docs, corpus)


if __name__ == "__main__":
    print("%.12f" % getidf("democracy"))
    print("%.12f" % getidf("foreign"))
    print("%.12f" % getidf("states"))
    print("%.12f" % getidf("honor"))
    print("%.12f" % getidf("great"))
    print("--------------")
    print("%.12f" % getweight("19_lincoln_1861.txt", "constitution"))
    print("%.12f" % getweight("23_hayes_1877.txt", "public"))
    print("%.12f" % getweight("25_cleveland_1885.txt", "citizen"))
    print("%.12f" % getweight("09_monroe_1821.txt", "revenue"))
    print("%.12f" % getweight("37_roosevelt_franklin_1933.txt", "leadership"))
    print("--------------")
    """
    print("(%s, %.12f)" % query("states laws"))
    print("(%s, %.12f)" % query("war offenses"))
    print("(%s, %.12f)" % query("british war"))
    print("(%s, %.12f)" % query("texas government"))
    print("(%s, %.12f)" % query("world civilization"))
    """
