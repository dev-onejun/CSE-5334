$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

#### KNN and

- Instance-based classifier
    - Find an instance in the training data that is similar to the new instance

    ex. Rote-learner: memorize all training data
        - Limitation
            -

    - Nearest Neighbor Classifiers
        - at training time, do nothing.

        - k가 짝수일 때, weighting을 통해 더 가까운 것으로 답을 내도록 한다. (ex. w = 1/distance) 제곱 해도 되고
            짝수가 아닐때도 사용한다. ?

        - k = 1일때, in order to make the computation faster, making a voronoi diagram during training so that we can find the nearest neighbor faster.

        - if k is too small, the classifier becomes sensitive to noise poitns. For example, if a new data is close to a noise point, the classifier will be affected by the noise point. although it gives another answer.
        - if k is too large, the accuracy of the classifier will be decreased.

        - Since Euclidean measure has limitations for the not normalized vectors, normalization is needed.

        * PEBLS
            - distance calculation
            - weighting algorithm
                - 실제로 제대로 classify했을 때, 맞는 sample에 weight 올리고, 틀린 sample에 weight 내리기















### References

$\tag*{}\label{n} \text{[n] }$
