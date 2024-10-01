from programming_assignment_1 import *

import pytest
import platform
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer


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


def test_getidf():
    idf = getidf("democracy")
    assert type(idf) == float


def test_compute_TF_IDFs():
    TF_IDFs = compute_TF_IDFs()
    assert type(TF_IDFs) == dict
    assert len(TF_IDFs) == 40
    for TF_IDF in TF_IDFs.values():
        norm = math.sqrt(sum(value**2 for value in TF_IDF.values()))
        assert pytest.approx(norm, 0.0001) == 1.0

    """
    def tokenizer(text):
        tokenizer = RegexpTokenizer(r"[a-zA-Z]+")
        return tokenizer.tokenize(text)

    vectorizer = TfidfVectorizer(
        smooth_idf=False,
        tokenizer=tokenizer,
        stop_words="english",
        min_df=-1,
    )
    temp_docs = [" ".join(doc) for doc in preprocessed_docs]
    tfidf_array = vectorizer.fit_transform(temp_docs).toarray()
    feature_names = vectorizer.get_feature_names_out()

    library_TF_IDFs = []
    for tfidf_per_doc in tfidf_array:
        library_TF_IDFs.append(
            {token: tfidf_per_doc[i] for i, token in enumerate(feature_names)}
        )

    def compare_tf_idf_dictionary(TF_IDFs, library_TF_IDFs) -> bool:
        for doc1, doc2 in zip(TF_IDFs, library_TF_IDFs):
            for key in doc1.keys():
                if doc1[key] != doc2[key]:  # ignore 0 values
                    print(f"key: {key}, doc1: {doc1[key]}, doc2: {doc2[key]}")
                    return False

        return True

    assert compare_tf_idf_dictionary(TF_IDFs, library_TF_IDFs) == True
    """


def test_getweight():
    pass


def test_query_vector():
    query_vector = get_query_vector("democracy")
    norm = math.sqrt(sum(value**2 for value in query_vector.values()))
    assert pytest.approx(norm, 0.0001) == 1.0


def test_query():
    pass
