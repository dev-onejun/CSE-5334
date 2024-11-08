$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

* Hierarchical Clustering
    * Agglomerative Clustering
        - start with individual points and a proximity matrix
        - merge the two closest points in the matrix
        - update the matrix from two points to one point column and repeat these steps
        ( Proxymixity (similarity) matrix는 대각석을 기준으로 symetric하기 때문에 왼쪽아래는 채울 필요 x)
        (distance matrix는 대각선을 기준으로 대칭이 아니기 때문에 왼쪽아래도 채워줘야 한다.)
            cf. 1 - similarity matrix = distance matrix

        * Similarity 종류 (matrix 값 update에만 쓰임. cluster를 merge할 때는 min distance (or max similar) 사용)
            - min
                - strength: circle base였던 k-means와 달리, hierarchical은 어떤 모양의 cluster도 만들 수 있다.
                - limitation: sensitive to noise and outliers (because it only consider the closest point)
            - max
                - **클러스터 간에 가장 먼 거리를 가진 점들끼리의 거리를 사용한다는거지,
                    cluster를 합칠 때는 그 min distances (or max similar) 중에 가장 작은 거리를 사용한다.**
                - strength: gives a upper bound of the dissimilarity (distance) among data
                            less sensitive to noise and outliers
                - limitation: tends to break large clusters
                => min은 ~일때, max는 ~일때 사용하면 좋다
            - group average
                - strength: less sensitive to noise and outliers
                - limitation:
            - Cluster Similarity: Ward's Method
                - strength: less sensitive to noise and outliers
                - limitation: biased towards globular clusters (원형 클러스터 만드는 경향이 있다?)
            - distance between centroids

            - average pairwise distance

        - cluster 내의 점들을 기준으로 distance (or similarity)를 계산한다는 게 포인트네

        - similarity든 distance든 둘 다 같은 Hierarchical cluster 결과를 만드는데,
            다른 점 하나는 dendrogram의 y 축의 순서가 다르다는 것이다. (0, 0.1, ... 1) vs (1, 0.9, ... 0)

        * K means와 다른 점
            - 한 점이 여러 클러스터에 속할 수 있다.
            - iteration이 반복되면서, kmeans는 데이터가 속한 클러스터가 바뀔 수 있지만, hierarchical은 한번 정해지면 바뀌지 않는다.

        * time complexity
            - K means: O ( n * k * I * d) where n: number of data, k: number of clusters, I: number of iterations while algorithms, d: number of data dimensions
            - Hierarchical: O(n^2 log n)
            => 보통 데이터가 다른 값에 비해 훨씬 크기 때문에, k means가 더 빠르다.
                - 많은 데이터에서는 hierarchical clustering를 사용하지 않는다















### References

$\tag*{}\label{n} \text{[n] }$
