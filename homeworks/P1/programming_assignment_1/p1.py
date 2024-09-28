"""
CSE-5334-DATA MINING
Professor: Dr. Marnim Galib

Writer: Wonjun Park
UTA ID: 1002237177
wxp7177@mavs.uta.edu

Programming Assignment 1
Fall 2024, Computer Science and Engineering, University of Texas at Arlington
"""

import os

CORPUS_ROOT = "./US_Inaugural_Addresses"


def read_files(CORPUS_ROOT) -> list:
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
    return docs
