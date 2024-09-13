$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

Similarity and Dissimilarity

Similarity is a measure of how much alike two data objects are. Dissimilarity is a measure of how different two data objects are.

Similarity normally normalized as [0,1] or [-1,1]. Dissimilarity is normalized as [0,1].

$$
\begin{array}{|c|c|c}}
\hline
& \text{Similarity} & \text{Dissimilarity \\
\hline
$$
ex. if G=2, S=1, B=0, then d(s,s) = |1 - 1| / 3-1
ex. s(G,B) = 1 - d(G,B)

The most common distance measure is Euclidean distance. The Euclidean distance between two points p and q is the length of the line segment connecting them.

$$
\text{dist} = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2} \\
\text{p and q are objects/records/instances VECTORS?} \\
\text{k is attributes}
$$

The standardization is necessary since, in most cases, the scale of each attribute is different. The standardization is done by subtracting the mean and dividing by the standard deviation.
or (target_d - min_d)/(max - min) ->[0,1]

Minkowski Distance is a generalization of Euclidean distance.

$$
\text{dist} = (\sum_{i=1}^{n} |p_i - q_i|^k)^{1/k} \\
\text{r = 1, Manhattan distance, L1 norm, r = 2, Euclidean distance, L2 norm, r = inf, Chebyshev distance (supremum distance?) L_max norm L_\inf norm}
$$
L_max norm converges to the maximum difference between the two vectors. (부연설명 필요)

Common Properties of a distance
- d(x,y) >= 0 (Positive definiteness)
- ...


Common Properties of a similarity
- s(p,q) = 1 only if p = q
..


Similarity between binary vectors

The vectors are binary, and they only can have the combinations 01, 10, 11, or, 00.

if we had a matrix
    anthony brutes Ceasar D E F
Anthony and cleopatry 1 0 1 0 1 1
Julius Ceasar 0 1 1 0 1 0

- Simple Matching Coefficient (SMC)
    SMC(A&C, JC) = 0.5
- Jaccard Coefficient
    JC(A&C, JC) = 0.4

Two coeeficients are only different in M_00 so that the meaning of it is important. In this example, 00 means the similarity so that SMC is betterh than Jaccard Coefficient. JAccard is more important when the data matrix has a lot of 00. In this case, SMC 근사한다 to 1 regardless of the interested information. As asking whether the object has a choice to be 0, which one, SMC or JACCARD, is the answer


Cosine Similarity


Pearson Correlation Coefficient

Pearson Correlation Coefficient is a measure of the linear correlation between two variables X and Y. It has a value between +1 and -1. +1 means that there is a perfect positive linear relationship (경향) between the two variables, -1 means that there is a perfect negative linear relationship between the two variables, and 0 means that there is no linear relationship between the two variables.


Quiz HINT

Ch3. what is 4Vs, data mining pipeline, the definition of data mining (what is, and what is not)
Ch4. what type of data is (nominal, ...) and (discrete, continuous). what is aggregation, sampling ,... and how to do it. Data quality issues (why happen, how to fix it).
Ch5. How to calculate similarity? Calculate Euclidean distance. If the data is not normalized, you should do noramalize. SMC and JAccard difference. Pearson Correlation Coefficient how to caculate it. (like draw matrix)


### References

$\tag*{}\label{n} \text{[n] }$
