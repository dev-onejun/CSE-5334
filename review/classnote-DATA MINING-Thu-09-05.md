$$
\begin{array}{|c|c|}
\hline
\text{Principle Component Analysis (PCA)} & \text{Singular Value Decomposition (SVD)} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

Types of Sampling
* Stratified sampling: from the Population, the ratio of each class is maintained in the sample.

Sample Size is matter since if the size is too small, not allowed to detect the pattern. If the size is too large, it is not efficient.
~비둘기집원리, if the number of classes is 10, the sample size shouold be at least 10 to select all the classes. 60개는 왜?

Curse of Dimensionality:
-> required Dimensionality Reduction
    - PCA: to find a projection that maximizes the variance of the data.
        Find the eigenvectors of the covariance matrix. the eigenvectors define the new space.
    - Feature Subset Selection
        - Brute-force approach
        - Embeded approach
        - Filter approach
        - Wrapper approach
    - Feature Creation
        - Feature Extraction with domain-specific knowledge
        - Mapping Data to a new space
        - Feature Construcction with combining features
            - ex. Combine signals with Fourier Transform or Wavelet Transform
    - Feature Discretization
        - Entropy-based Discretization (using Class Labels)
        (w/o class labels)
        - Equal-width Discretization
        - Equal-frequency Discretization
        - Clustering-based Discretization(K-means)

    Attribute Transformation?

### References

$\tag*{}\label{n} \text{[n] }$
