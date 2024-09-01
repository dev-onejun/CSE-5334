$$
\begin{array}{|c|c|}
\hline
\text{Knowledge Discovery (KDD)} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

**Next Thu. Sep. 5th EXAM**
<!--정리하고, 정리해라-->

* tf-raw
* tf-weighted (log-tf)
* df
* idf


**Cosine Similiarity**

$$
\cos
$$

Whatever preprocessing do in query, do in document.

Different weightings for queries and documents are expressed as ddd.qqq format. For exmaple, 'lnc.ltn' for each character is in lecture note 60 page.

|term|tf|logtf|df|idf|tf-idf| |tf|logtf|idf|tf-idf|norm tf-idf|
|---|---|---|---|---|---|---|---|---|---|---|---|
|best|1|1|50000|1.3|1.3| |0|0|1|0|0|
|car|1|1|10000|2|2| |1|1|1|1|0.52|
|insurance|1|1|1000|3|3| |2|1.3|1|1.3|0.68|
|auto|0|0|5000|3.3|0| |1|1|1|1|0.52|

Q. Isn't it bad to not idf-weight the document?
A. IT depends. some cases


CH2. **Data Mining**
Four 'V's in Big data
1. volume
- the size of the datasets

2. Variety
- Types of data
    - Structured data
        ex. Relational Database Tables, CSV/TSV files
    - Semi-structured data
        ex. XML, JSON, RDF
    - Unstructed data
        text data, videos, audio, binary data files, ..

3. Velocity

4. Veracity

Q. What is Data mining
A. Extraction of interesting patterns or knowledge from huge amount of data.

KDD Process
Data -> Target Data -> ...
    selection


### References

$\tag*{}\label{n} \text{[n] }$
