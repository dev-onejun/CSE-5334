$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

Programming Assignment 오래 걸릴거니까 빨리 시작해라 ,,
- TF-idf 계산식 직접 만들라는건가? 가져다 쓰지말고?
    - log-weighted tf and log-weighted idf are used
    - The TF-IDF vectors should be normalized, resulting in the length of each vector being 1.
        - TIP: 조금 해보고 맞는지 확인해라. -> 라이브러리로 확인하고 하면 될듯?
- The given guideline preprocessed 'lowercase', 'tokenization', 'stopword_removal', and 'stemming'. Those preprocessing steps need to apply in both query and document.
- ltc.lnc weighting scheme
- 6까지는 수업에서 한 거랑 같음
- 7 cosine sim calculate method
    - documents가 billion이면 다 계산 힘듬. 실제로 쓸 수 있는 방법이 있어야 함
        - posting list? **내용 맞는지는 확인해봐야함..**
            - 각 term에 대해 top k 리스트를 저장?
                - query가 들어왔을 때 그 term에 대한 posting list를 가져와서 계산
                - query에 2개 term이 있으면, in the worst case, k + k 개의 documents 계산. in the best K개.

                - k+k개에서 documents 종류가 다르면 query의 term에 대해 값을 가지고 있지 않을 수 있음
                    - top k로 sorting한 것이므로, 없는 term의 가장낮은 값으로 대체해 사용하면 됨. 그 값은 해당 term이 없는 문서가 가질 수 있는 최대값임(sorting한 것이므로)
                    - 단 false score라고 표시하는듯?

                - 이후 tf-idf score 랭킹에서 true이면 그대로 사용하면 되고,
                    false이면 없는 term list에 대해 fetch more해서 랭킹 갱신하면 됨






---

TREE. Gini Index

To decide the best attribute to split,
Select an attribute which make the highest gain.

$$
gain = gini(parent) - (weighted average)gini(children)
$$

For continuous attributes, to determine the split point, converting it into discrete values is needed.
- 값의 중간에 있는 것을 threshold로 사용, n+1개의 threshold가 n개의 값에 대해 생김
- 각 threshold에 대해 gini index를 계산하고, 가장 작은 것을 선택
- 그러나, 모든 threshold에 대해 계산하는 것은 비효율적이므로, mean,median 같은 값을 사용하기도 함


after splitting, the gain for each attribute is needed to be recalculated.


two split보다 three가 쉬우니? 시험에선 three가능하게 할 것?






### References

$\tag*{}\label{n} \text{[n] }$
