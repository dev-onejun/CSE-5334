$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

* Association Rule Mining
    - Itemset: a collection of one or more items
        ex. {milk, bread, butter} : 3-itemset
    - Support Count: the number of transactions that contain a particular itemset
        ex. from
        $$
        \begin{array}{cc}
        \text{Transaction ID} & \text{Items} \\
        \hline
        1 & \text{Bread, Milk} \\
        2 & \text{Bread, Diaper, Beer, Eggs} \\
        3 & \text{Milk, Diaper, Beer, Coke} \\
        4 & \text{Bread, Milk, Diaper, Beer} \\
        5 & \text{Bread, Milk, Diaper, Coke} \\
        \end{array}
        $$
        $\to \sigma ( \{milk, bread, butter\} ) = 2$
        - minimum value: 0?
    - Support: the fraction of transactions that contain a particular itemset
        ex. $s( \{milk, bread, butter\} ) = \frac{2}{5}$
    - Frequent Itemset: an itemset whose support is greater than or equal to a minsup threshold

    - Association Rule: an implication expression of the form $X \to Y$ where $X$ and $Y$ are disjoint? itemsets
        - ex. {milk, bread} $\to$ {butter}
        - X 와 Y는 emptyset이 될 수 없다.
    - Rule Evaluation Metrics: Support (s), Confidence (c)
        - asosciation rule 순서 (x를 y로, y를 x로)를 바꾸면, support 값은 같지만 confidence 값은 다르다.

    - The goal of association rule mining is to find all rules having
        - support $\geq$ minsup
        - confidence $\geq$ minconf

    * Brute-Force Approach
        - Generate all possible association rules ($2^n - 2$ where $n$ is the number of items, 2: emptyset)
        - Calculate support and confidence for each rule
        - Prune rules that do not meet minsup and minconf
        - $\to$ computationally expensive (The number of possible rules increases exponentially with the number of items in the dataset)
















### References

$\tag*{}\label{n} \text{[n] }$
