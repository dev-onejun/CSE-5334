$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

* Problems with selecting initial centroids
- The chance to select well-separated initial centroids is

$$
P = \frac{\text{number of ways to select one centroid from each cluster}}{\text{number of ways to select k centroids from n points}} = \frac{k! n^k}{(kn)^k} = \frac{k!}{(k)^k}
$$

the probability becomes low if k bigger

    - solution
        - multiple runs. but it is not efficient and do not guarantee the best solution
        - sample and use hierarchical clustering to find initial centroids ??
        - select more than k initial centroids and then select the most widely separated among these initial centroids
        - postprocessing
            - select outliers ?
        - bisecting K-means
            ?

* Handling empty clusters
    - solution
        -


* Updating centers incrementally
- convergence more faster, never get an empty cluster
- but more expensive and have an order dependency (어떤 포인트를 먼저 계산하느냐에 따라 결과가 달라질 수 있음)

* possible pre-processing
    - normalization (since distance)
    - eliminate outliers (due to making centroids far away)
* possible post-processing
    - eliminate samller clusters
    - split loose clusters (dense한 cluster가 두 개 이상일 수 있기 때문에)
    - merge cluster between two clusters that have relatively low SSE (cluser가 너무 많을떄)
        merge후 전체 SSE는 높아짐


* Bisecting K-means
- select the cluster with the highest SSE and split it into two clusters
    (make the entire SSE smaller)
- repeat until the number of clusters is k
- 2개씩 나누면서 계층구조 만들어짐

- more computation but give better results than k-means

* limitations of k-means
    - size
        - original cluster의 원형 크기(반지름)이 서로 달랐어도, k-means는 모두 같은 크기로 계산
    - Densities
        - original cluster의 밀도가 다르면, SSE가 original 보다 낮은 다른 형태의 클러스터를 찾을 수도 있음
    - non-globular shapes. ex. 별 모양의 클러스터는 찾지 못함. 원형 클러스터로 계산하기 때문에

    - Solution
        - 원래보다 더 많은 k개의 클러스터를 찾은다음 post-processing으로 합치거나, split


* K값 선택 방법
- SSE의 threshold 선택해놓고 (elbow method) 거기에 최초 도달하는 K값을 선택. (k값이 커질수록 SSE는 줄어들기 때문에)


##### Hierarchical clustering

-
- 클러스탁 몇 개인지 추측할 필요 없음

* Agglomerative hierarchical clustering
* Divisive hierarchical clustering










### References

$\tag*{}\label{n} \text{[n] }$
