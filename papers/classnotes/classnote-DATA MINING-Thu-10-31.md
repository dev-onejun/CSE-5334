$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

* idea behind clustering
    - group similar objects together

* Types of Clusterings
- Hierarchical clustering
    - a set of nested clusters organized as a hierarchical tree
        ex. 동물 분류 (포유동물)
- Partitional clustering
    - divides each data into non-overlapping subsets

* Distinctions between sets of clusters
    - exclusive vs. non-exclusive
    - fuzzy vs. non-fuzzy
    - partial vs. complete
    - heterogeneous vs. homogeneous

* Types of Clusters
    - Well-separated clusters
    - Center-based clusters
        - centroid or medoid

            - well-separated에서는 항상 inter-cluster distance가 크다.
            - center-based에서는 intra-cluster distance가 작을 수 있다.

    - Contiguous clusters
    - Density-based clusters
    - Conceptual (Shared Property) clusters


    * intra-cluster: distance between objects in the same cluster
    * inter-cluster: distance between objects in different clusters


* Objective Function in clusters
    - NP hard
    -


...

* Clustering Algorithms
- K-means
    - partitional clustering approach
    - each cluster is associated with a centroid
    - each point is assigned to the cluster with the closest centroid
    - K must be specified
        - to evaluate, multiple k values are tested

    - Choosing initial centroids is important since it affects the final clusters

    - Detail algorithms ...

    - Evaluate method
        intial centroids가 중요하다면, 어떤 cluster가 좋은지 어떻게 판단할 것인가?
        - SSE (Sum of Squared Error)
            - 작을수록 좋음.



- Hierarchical clustering
















### References

$\tag*{}\label{n} \text{[n] }$
