$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

#### Naive Bayes Classifier

Bayes theorem is a way to calculate the probability of a hypothesis given our prior knowledge. The theorem is stated as follows:

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

Based on Bayes theorem with strong assumptions that the attributes are conditionally independent, Naive Bayes classifier is a simple probabilistic classifier with attributes that are conditionally independent given the class label. When each attribute is defined as $A_1, A_2, \ldots, A_n$ and the class label is $C$, the Naive Bayes classifier is defined as:

$$
\text{Naive Bayes Classifier} = P(C \mid A_1, A_2, \ldots, A_n)
\begin{cases} P(C \mid A_1, A_2, \ldots, A_n) = \frac{P(A_1, A_2, \ldots, A_n \mid C)P(C)}{P(A_1, A_2, \ldots, A_n)} \\
P(A_1, A_2, \ldots, A_n \mid C) = P(A_1 \mid C)P(A_2 \mid C) \ldots P(A_n \mid C)
\end{cases}
$$
Hidden Markov Model의 1차 가정?이었던 거같은데

For continuous attributes, the probability density function is used to estimate the probability of the attribute given the class label, assuming that the attributes follow a normal distribution.

In example in p.9, if P(Income=120 | Yes) = 아주 작게 나올 것이다. YES에서 120이 나올 확률은 정규분포에서 매우 작기 때문

어떤 attribute에 대한 조건부확률이 0이 되면, 전체 확률이 0이 되어버리는 문제가 발생한다. naive bayes limitation.

To solve this, Laplace smoothing is used to avoid zero probabilities. Laplace smoothing is a technique to add a small value to the probability of each attribute to avoid zero probabilities. it add 1 to the numerator and the number of classes to the denominator. 참고로 이를 일반화한게 m-estimate이다.

In sumary, Naive Bayes classifier is robust to noise data and irrelevant attribute, easy to handle missing values by ignoring the missing values, and


### References

$\tag*{}\label{n} \text{[n] }$
