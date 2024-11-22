$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

##### Frequent Item Generation Problem

Brute-force approach: too costly.

---

* Minimum Support Value: if the value is set as low, the number of frequent itemsets will be high as well as if the value is set as high, the number of frequent itemsets will be low **THRESHOLD**

* Apriori principle: If an itemset is frequent, then all of its subsets must also be frequent.
    - Support of an itemset never exceeds the support of its subsets.
    ex. if Support Count ({M}) = 300, then Support Count ({M, P}) $\leq$ 300
        - tree의 아래쪽으로는 infrequent임을, 위로는 frequent임을 알 수 있다

* Apriori Algorithm
    - 가능한 조합을 모두 만든는 게 아니라, apriori principle에 따라 single itemset부터 min_sup를 절대 만족할 수 없는 item들을 지워가면서 가능한 조합을 만들어나간다.
    - O(m $\times$ n) ?

    * Generate Hash Tree ?
        - Big-o 줄일 수 있다?


















### References

$\tag*{}\label{n} \text{[n] }$
