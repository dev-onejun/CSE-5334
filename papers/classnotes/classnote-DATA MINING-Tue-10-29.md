$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

- Test of Significance
    Suppose a circumstance that
        M1 got 85% accuracy but tested only on 30 instances
        M2 got 75% accuracy tested on 5000 instances

    - 모델들이 테스트된 데이터셋이 다를 때 (특히 숫자가 다를 때) 통계적으로 유의미하게 비교할 수 있는 방법이다


    - Statistically significant
        - 신뢰구간을 통해 모델들을 비교, ex. 95% 신뢰도로 ()사이에 accuracy가 있다.
        - 어떤 모델이 다른 모델의 신뢰구간을 벗어나있을 때에만, 그 모델이 더 좋다 나쁘다를 말할 수 있다.

        - each model follows binominal distribution(the number of samples, accuracy)

    - In the bell curve, roughly 68% of the data is within 1 standard deviation, 95% is within 2 standard deviations, and 99.7% is within 3 standard deviations.

    - 시험에서, z-table이나 confidene interavl for p의 식은 주어질 것임

    - the more the number of samples increases, the more the confidence interval narrows down.

    - Error rate also follows binominal distribution (the number of samples, error rate)
        - ?

* For Programming Assignment 2,
    - hyperparmeter 알아서 최고로 찾아서 하기
        ex. k value in KNN, depth of decision tree

    - 통계정보가지고, 농구선수의 포지션(슈팅가드 등) 맞추기
        - 20% baseline accuracy (5개이므로)

        + set random_state = 0


* Clustering
    - Infra-cluster
    - Inter-cluster

    - Types of clusterings
        - Partitional Clustering
        - Hierarchical Clustering













### References

$\tag*{}\label{n} \text{[n] }$
