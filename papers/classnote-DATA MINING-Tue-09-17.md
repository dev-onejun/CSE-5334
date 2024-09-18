$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

classification \in prediction

The best split in a decision tree is that minimizes the impurity of the child nodes. (in page 30, car type makes only 2 false in the child nodes)

The impurity of a node is measured by 1) Gini index, 2) Entropy, 3) Misclassification error.

**1) Gini index**

$$
Gini = 1 - \sum_{i=1}^{c} p_i^2
$$

$P(j|t)$ is the relative frequency of class $j$ at node $t$. For example, (node t - c_1: 5, c_2: 3) P(c1|t) = 5/8 = 0.625, P(c2|t) = 3/8 = 0.375
GINI = 1 - {P(c_1t)^2 + P(c2|t)^2} = 30/64

The lowest GINI is 0, which means the node is pure. The highest GINI for 2 classes is 0.5 when the node is equally distributed.
For 3 classes, the minimum is 0 and the maximum is 0.67
In general, when $n_e$ = # of classes, minimum = 0, max = 1 - (1/n_c)

If the ratio of the classes are same, the GINI value is same regardless of the number of datas.

Computing GINI index (Binary Attributes)
GINI - GINI(Children) aND GAIN ..

(FOR QUIZ TIP, 직관적으로 ratio가 (2개 클래스에서) 1:1에 가까우면 0.5에 가깝다고 생각하면 됨)

### References

$\tag*{}\label{n} \text{[n] }$
