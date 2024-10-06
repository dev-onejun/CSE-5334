import pytest
import platform

import nltk
from nltk.corpus import stopwords

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

from programming_assignment_1 import *


def test_readFiles():
    assert type(read_files(CORPUS_ROOT)) == dict
    assert len(read_files(CORPUS_ROOT)) == 40


def test_tokenize():
    tokenized_docs, whole_tokens = tokenize(read_files(CORPUS_ROOT))
    assert type(tokenized_docs) == dict
    assert type(whole_tokens) == list


def test_preprocess():
    # Exception handling if the test runs on the remote machine named "runner"
    try:
        tokenized_docs, tokens_set = tokenize(read_files(CORPUS_ROOT))
        preprocessed_docs, corpus = preprocess(tokenized_docs, tokens_set)

        assert type(corpus) == list
        assert (
            stopwords.words("english") not in corpus
        )  # Check remove_stopwords function

        _, tokens = tokenize(read_files(CORPUS_ROOT))
        STOPWORDS = stopwords.words("english")
        remove_stopwords_tokens = [token for token in tokens if token not in STOPWORDS]

        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in remove_stopwords_tokens]
        assert corpus == stemmed_tokens
    except LookupError:
        print("LookupError: Resource 'corpora/stopwords' not found.")
        if platform.node() == "runner":
            print(
                "The test function is passed because it runs on the remote machine named 'runner'"
            )
            pass


test_docs = {
    "doc1": "cse students",
    "doc2": "cse students uta cse",
}
test_tokenized_docs, test_tokens_set = tokenize(test_docs)
test_preprocessed_docs, test_corpus = preprocess(test_tokenized_docs, test_tokens_set)


def test_compute_idf():
    N = len(test_preprocessed_docs)

    assert compute_idf(N, test_preprocessed_docs, "cse") == 0
    assert compute_idf(N, test_preprocessed_docs, "students") == 0
    assert compute_idf(N, test_preprocessed_docs, "uta") == log(2, 10)


def test_compute_TF_IDFs():
    test_TF_IDFs, test_IDF_vectors = compute_TF_IDFs(
        test_preprocessed_docs, test_corpus
    )
    print(test_TF_IDFs)

    assert test_TF_IDFs["doc1"]["cse"] == 0.0
    assert test_TF_IDFs["doc1"]["student"] == 0.0
    assert test_TF_IDFs["doc2"]["cse"] == 0.0
    assert test_TF_IDFs["doc2"]["student"] == 0.0
    assert (
        test_TF_IDFs["doc2"]["uta"]
        == ((1 + log(1, 10)) * test_IDF_vectors["uta"]) / test_IDF_vectors["uta"]
    )


def test_getidf():
    print("%.12f" % getidf("democracy"))
    print("%.12f" % getidf("foreign"))
    print("%.12f" % getidf("states"))
    print("%.12f" % getidf("honor"))
    print("%.12f" % getidf("great"))


def test_getweight():
    print("%.12f" % getweight("19_lincoln_1861.txt", "constitution"))
    print("%.12f" % getweight("23_hayes_1877.txt", "public"))
    print("%.12f" % getweight("25_cleveland_1885.txt", "citizen"))
    print("%.12f" % getweight("09_monroe_1821.txt", "revenue"))
    print("%.12f" % getweight("37_roosevelt_franklin_1933.txt", "leadership"))


def test_get_query_vector():
    test_query = "uta uta department"
    test_query_vector = get_query_vector(test_query)

    EXPECTED_VECTOR_LENGTH = math.sqrt((1 + log(2, 10)) ** 2 + (1 + log(1, 10)) ** 2)
    print(EXPECTED_VECTOR_LENGTH)
    EXPECTED_VECTOR = {
        "uta": (1 + log(2, 10)) / EXPECTED_VECTOR_LENGTH,
        "department": (1 + log(1, 10)) / EXPECTED_VECTOR_LENGTH,
    }

    for computed_value, expected_value in zip(
        test_query_vector.values(), EXPECTED_VECTOR.values()
    ):
        assert computed_value == expected_value


def test_query():
    pass
