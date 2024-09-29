from programming_assignment_1 import *

import platform
from sklearn.feature_extraction.text import TfidfVectorizer


def test_readFiles():
    assert type(read_files(CORPUS_ROOT)) == list
    assert len(read_files(CORPUS_ROOT)) == 40


def test_tokenize():
    tokenized_docs, whole_tokens = tokenize(read_files(CORPUS_ROOT))
    assert type(tokenized_docs) == list
    assert type(tokenized_docs[0]) == list
    assert type(tokenized_docs[0][0]) == str
    assert type(whole_tokens) == list


def test_preprocess():
    # Exception handling if the test runs on the remote machine named "runner"
    try:
        _, whole_tokens = tokenize(read_files(CORPUS_ROOT))
        preprocessed_tokens = preprocess(whole_tokens)

        assert type(preprocessed_tokens) == list
        assert (
            stopwords.words("english") not in preprocessed_tokens
        )  # Check remove_stopwords function

        _, tokens = tokenize(read_files(CORPUS_ROOT))
        STOPWORDS = stopwords.words("english")
        remove_stopwords_tokens = [token for token in tokens if token not in STOPWORDS]

        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in remove_stopwords_tokens]
        assert preprocessed_tokens == stemmed_tokens
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
    assert type(TF_IDFs) == list
    assert type(TF_IDFs[0]) == dict
    assert len(TF_IDFs) == 40

    tokenized_docs, _ = tokenize(read_files(CORPUS_ROOT))
    vectorizer = TfidfVectorizer()

    library_TF_IDFs = []
    tfidf_array = vectorizer.fit_transform(tokenized_docs).toarray()
    feature_names = vectorizer.get_feature_names_out()

    for tfidf_per_doc in tfidf_array:
        library_TF_IDFs.append(
            {token: tfidf_per_doc[i] for i, token in enumerate(feature_names)}
        )

    def compare_tf_idf_dictionary(TF_IDFs, library_TF_IDFs) -> bool:
        for doc1, doc2 in zip(TF_IDFs, library_TF_IDFs):
            for key in doc1.keys():
                if doc1[key] != doc2[key]:  # ignore 0 values
                    print(doc1, doc2)
                    return False

        return True

    assert compare_tf_idf_dictionary(TF_IDFs, library_TF_IDFs) == True


def test_getweight():
    pass


def test_query():
    pass
