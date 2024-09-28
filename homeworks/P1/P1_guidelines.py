# %% [markdown]
# # CSE 5334 Programming Assignment 1 (P1)

# %% [markdown]
# In this assignment, you will implement a toy "search engine" in Python. You code will read a corpus and produce TF-IDF vectors for documents in the corpus. Then, given a query string, you code will return the query answer--the document with the highest cosine similarity score for the query. 
# 
# The instructions on this assignment are written in an .ipynb file. You can use the following commands to install the Jupyter notebook viewer. You can use the following commands to install the Jupyter notebook viewer. "pip" is a command for installing Python packages. You are required to use Python 3.5.1 or more recent versions of Python in this project. 
# 
#     pip install jupyter
# 
#     pip install notebook (You might have to use "sudo" if you are installing them at system level)
# 
# To run the Jupyter notebook viewer, use the following command:
# 
#     jupyter notebook P1.ipynb
# 
# The above command will start a webservice at http://localhost:8888/ and display the instructions in the '.ipynb' file.

# %% [markdown]
# ### Requirements

# %% [markdown]
# * This assignment must be done individually. You must implement the whole assignment by yourself. Academic dishonety will have serious consequences.
# * You can discuss topics related to the assignment with your fellow students. But you are not allowed to discuss/share your solution and code.

# %% [markdown]
# ### Dataset

# %% [markdown]
# We use a corpus of 40 Inaugural addresses of different US presidents. We processed the corpus and provided you a .zip file, which includes 40 .txt files.

# %% [markdown]
# ### Programming Language

# %% [markdown]
# 1. You are required to submit a single .py file of your code.
# 
# 2. You are expected to use several modules in NLTK--a natural language processing toolkit for Python. NLTK doesn't come with Python by default. You need to install it and "import" it in your .py file. NLTK's website (http://www.nltk.org/index.html) provides a lot of useful information, including a book http://www.nltk.org/book/, as well as installation instructions (http://www.nltk.org/install.html).
# 
# 3. In programming assignment 1, other than NLTK, you are not allowed to use any other non-standard Python package. However, you are free to use anything from the the Python Standard Library that comes with Python (https://docs.python.org/3/library/).

# %% [markdown]
# ### Tasks

# %% [markdown]
# You code should accomplish the following tasks:
# 
# (1) <b>Read</b> the 40 .txt files, each of which has the transcript of inaugural addresses by different US presidents. The following code does it. Make sure to replace "corpusroot" by your directory where the files are stored. In the example below, "corpusroot" is a sub-folder named "US_Inaugural_Addresses" in the folder containing the python file of the code. 
# 
# In this assignment we ignore the difference between lower and upper cases. So convert the text to lower case before you do anything else with the text. For a query, also convert it to lower case before you answer the query. 

# %%
import os
corpusroot = './US_Inaugural_Addresses'
for filename in os.listdir(corpusroot):
    if filename.startswith('0') or filename.startswith('1'):
        file = open(os.path.join(corpusroot, filename), "r", encoding='windows-1252')
        doc = file.read()
        file.close() 
        doc = doc.lower()

# %% [markdown]
# (2) <b>Tokenize</b> the content of each file. For this, you need a tokenizer. For example, the following piece of code uses a regular expression tokenizer to return all course numbers in a string. Play with it and edit it. You can change the regular expression and the string to observe different output results. 
# 
# For tokenizing the inaugural Presidential speeches, we will use RegexpTokenizer(r'[a-zA-Z]+')
# 

# %%
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[A-Z]{2,3}[1-9][0-9]{3,3}')
tokens = tokenizer.tokenize("CSE4334 and CSE5534 are taught together. IE3013 is an undergraduate course.")
print(tokens)

# %% [markdown]
# (3) Perform <b>stopword removal</b> on the obtained tokens. NLTK already comes with a stopword list, as a corpus in the "NLTK Data" (http://www.nltk.org/nltk_data/). You need to install this corpus. Follow the instructions at http://www.nltk.org/data.html. You can also find the instruction in this book: http://www.nltk.org/book/ch01.html (Section 1.2 Getting Started with NLTK). Basically, use the following statements in Python interpreter. A pop-up window will appear. Click "Corpora" and choose "stopwords" from the list.

# %%
import nltk
nltk.download()

# %% [markdown]
# After the stopword list is downloaded, you will find a file "english" in folder nltk_data/corpora/stopwords, where folder nltk_data is the download directory in the step above. The file contains 179 stopwords. nltk.corpus.stopwords will give you this list of stopwords. Try the following piece of code.

# %%
from nltk.corpus import stopwords
print(stopwords.words('english'))

# %% [markdown]
# (4) Also perform <b>stemming</b> on the obtained tokens. NLTK comes with a Porter stemmer. Try the following code and learn how to use the stemmer.

# %%
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem('studying'))
print(stemmer.stem('vector'))
print(stemmer.stem('entropy'))
print(stemmer.stem('hispanic'))
print(stemmer.stem('ambassador'))

# %% [markdown]
# (5) Using the tokens, compute the <b>TF-IDF vector</b> for each document. Use the following equation that we learned in the lectures to calculate the term weights, in which $t$ is a token and $d$ is a document:  $$w_{t,d} = (1+log_{10}{tf_{t,d}})\times(log_{10}{\frac{N}{df_t}}).$$ Note that the TF-IDF vectors should be normalized (i.e., their lengths should be 1). 
# 
# Represent a TF-IDF vector by a dictionary. 

# %% [markdown]
# (6) Given a query string, calculate the query vector. (Remember to convert it to lower case.) In calculating the query vector, don't consider IDF. I.e., use the following equation to calculate the term weights in the query vector, in which $t$ is a token and $q$ is the query: $$w_{t,q} = (1+log_{10}{tf_{t,q}}).$$
# The vector should also be normalized. 

# %% [markdown]
# (7) Find the document that attains the highest <b>cosine similarity</b> score. If we compute the cosine similarity between the query vector and every document vector, it is too inefficient. Instead, implement the following method:
# 
# (7.1) For each token $t$ that exists in the corpus, construct its <b>postings list</b>---a sorted list in which each element is in the form of (document $d$, TF-IDF weight $w$). Such an element provides $t$'s weight $w$ in document $d$. The elements in the list are sorted by weights in descending order. 
# 
# (7.2) For each token $t$ in the query, return the top-10 elements in its corresponding postings list. If the token $t$ doesn't exist in the corpus, ignore it. 
# 
# (7.3) If a document $d$ appears in the top-10 elements of every query token, calculate $d$'s cosine similarity score. Recall that the score is defined as follows. Since $d$ appears in top-10 of all query tokens, we have all the information to calculate its actual score $sim(q,d)$.
# 
# $$ sim(q,d) = \vec{q} \cdot \vec{d} = \sum_{t\ \text{in both q and d}} w_{t,q} \times w_{t,d}.$$
# 
# (7.4) If a document $d$ doesn't appear in the top-10 elements of some query token $t$, use the weight in the 10th element as the upper-bound on $t$'s weight in $d$'s vector. Hence, we can calculate the upper-bound score for $d$ using the query tokens' actual and upper-bound weights with respect to $d$'s vector, as follows. 
# 
# $$ \overline{sim(q,d)} = \sum_{t \in T_1} w_{t,q} \times w_{t,d} + \sum_{t\in T_2} w_{t,q} \times \overline{w_{t,d}}.$$
# 
# In the above equation, $T_1$ includes query tokens whose top-10 elements contain $d$. $T_2$ includes query tokens whose top-10 elements do not contain $d$. $\overline{w_{t,d}}$ is the weight in the 10-th element of $t$'s postings list. As a special case, for a document $d$ that doesn't appear in the top-10 elements of any query token $t$, its upper-bound score is thus: 
# 
# $$ \overline{sim(q,d)} = \sum_{t\in q} w_{t,q} \times \overline{w_{t,d}}.$$
# 
# (7.5) If a document's actual score is better than or equal to the actual scores and upper-bound scores of all other documents, it is returned as the query answer. 
# 
# If there isn't such a document, it means we need to go deeper than 10 elements into the postings list of each query token. 

# %% [markdown]
# ### What to Submit 

# %% [markdown]
# <b> Submit through Canvas your source code in a single .py file.</b> You can use any standard Python library. The only non-standard library/package allowed for this assignment is NLTK. You .py file must define at least the following functions:
# 
# * getidf(token): return the inverse document frequency of a token. If the token doesn't exist in the corpus, return -1. You should stem the parameter 'token' before calculating the idf score.
# 
# * getweight(filename,token): return the normalized TF-IDF weight of a token in the document named 'filename'. If the token doesn't exist in the document, return 0. You should stem the parameter 'token' before calculating the tf-idf score.
# 
# * query(qstring): return a tuple in the form of (filename of the document, score), where the document is the query answer according to 7.5. If no document contains any token in the query, return ("None",0). If we need more than 10 elements from each posting list, return ("fetch more",0).

# %%
print("%.12f" % getidf('democracy'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('states'))
print("%.12f" % getidf('honor'))
print("%.12f" % getidf('great'))
print("--------------")
print("%.12f" % getweight('19_lincoln_1861.txt','constitution'))
print("%.12f" % getweight('23_hayes_1877.txt','public'))
print("%.12f" % getweight('25_cleveland_1885.txt','citizen'))
print("%.12f" % getweight('09_monroe_1821.txt','revenue'))
print("%.12f" % getweight('37_roosevelt_franklin_1933.txt','leadership'))
print("--------------")
print("(%s, %.12f)" % query("states laws"))
print("(%s, %.12f)" % query("war offenses"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("texas government"))
print("(%s, %.12f)" % query("world civilization"))

# %% [markdown]
# ### Evaluation

# %% [markdown]
# Your program will be evaluated using the following criteria: 
# 
# * Correctness (75 Points)
# 
# We will evaluate your code by calling the functions specificed above (getidf - 20 points; getweight - 25 points; query - 30 points). So, make sure to use the same function names, parameter names/types/orders as specified above. We will use the above test cases and other queries and tokens to test your program.
# 
# 
# * Preprocessing, Efficiency, modularity (25 Points)
# 
# You should correctly follow the preprocessing steps. An efficient solution should be able to answer a query in a few seconds, you will get deductions if you code takes too long to run (>1 minute). Also, it should consider the boundary cases. Your program should behave correctly under special cases and even incorrect input. Follow good coding standards to make your program easy to understand by others and easy to maintain/extend. 
# 


