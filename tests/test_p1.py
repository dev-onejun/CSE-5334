from programming_assignment_1 import *


def test_readFiles():
    assert type(read_files(CORPUS_ROOT)) == list
    assert len(read_files(CORPUS_ROOT)) == 40
