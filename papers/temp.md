$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\end{array}
$$


Among a few algorithms like Hunt's Algorithm, CART, ID3, C4.5, SLIQ, and SPRINT, the paper reviewed Hunt's Algorithm, a one of the earliest decision tree algorithms and the foundation of other decision tree algorithms.

**Hunt's Algorithm** is a recursive algorithm that partitions the data into subsets based on the attribute value, ensuring that each partition is as pure as possible. The subset is considered pure if all the data in the subset belongs to the same class or no further meaningful partition can be made. The algorithm is as follows:

``` plaintext
Start from a tree with a single root node containing all the training data.

Recursive:
    1. Check if the node is homogeneous (pure).
        - If true, make the node a leaf node and label it with the class. END the branch.
        - If not, continue to the next step.
    2. Check if the node is empty.
        - If true, make the node a leaf node. END the branch.
        - If not, continue to the next step.
    3. Check whether the node has conflicting data, a same label with different values.
        - If true, mark the node as a leaf node. END the branch.
        - If not, continue to the next step.
    4. Split the node into child nodes based on the attribute.
        - During the split, the algorithm calculates the impurity of the child nodes using the 1) Gini index, 2) entropy, or 3) misclassification error.
        - After splitting, the Gain is recalculated to renew the tree state.

Terminate:
    1. If all nodes become leaf nodes during the recursive process.
    2. If the split does not show certain improvement set beforehand threshold, regarding the impurity
        - When it comes to the large number of data, the accuracy would not be improved significantly even after the split.
        - For example, if 100,000,000 data were input, in a certain point such as a 1,000 impurity node, the split would not impact to the accuracy of the whole tree even if the half of the data were wrong.
        - This also leads the tree to be overfitted.
```

Regarding the split, two questions are raised. **1)** How to split with different types of attributes? **2)** How to determine the best split?

The first question is answered by the type of attributes. Nominal and Ordinal attributes are treated as categorical attributes. For example, if the attribute is color, the node is split into red, blue, and green or if the attribute is size, the node is split into small, medium, and large. In this case, if categories are grouped into two, this is called 2-way split. If categories are grouped into three or more, this is called multi-way split. Interval and Ratio attributes are treated as numeric attributes. For example, if the attribute is age, the node is split into age < 20, 20 <= age < 40, 40 <= age < 60, and age >= 60. Aslike the categorical attributes, if the numeric attributes are grouped into two, this is called binary split. If the numeric attributes are grouped into three or more, this is called multi-split.

To determine whether 2-way or multi-way split is needed, the algorithm calculates **Gain** with the following metrics for each possible children node. Especially for the continuous attributes, the algorithm converts the continuous values into discrete values the threshold usually the mean or median because it is inefficient that the algorithm calculates the metrics for all possible threshold values. (In the class, only 2-way split will be used.)

The second question is dealt with the metrics to measure the impurity of the child nodes. Three evaluation metrics are used to measure the impurity of the child nodes. **1) Misclassification error** is the simplest

However, the metric

**2) Gini index**, also known as Gini impurity, is a measure to quantify the impurity or diversity of a dataset. The formula for calculating the Gini index is:

$$
Gini = 1 - \sum_{i=1}^{c} p_i^2 \
\left\{
\begin{aligned}
& c = \text{the number of classes} \\
& p_i = \text{the probability of selecting an item of class } i \text{ in the node}
\end{aligned}
\right.
$$

Two properties, regarding min-max and the ratio of the classes, are notable. **1)** The loweset value of Gini index is 0, when the node is pure. The highest value of Gini index is 0.5 for the number of classes is 2 and 0.67 for 3 classes when the node is equally distributed. In general, when the number of classes is $n_c$, $Gini = \left\{ \begin{aligned} & \text{minimum} = 0 \\ & \text{maximum} = 1 - \frac{1}{n_c} \end{aligned} \right.$ **2)** If the ratio of the classes are same between different nodes, the Gini value is same regardless of the number of datas. These properties lead the intuition to estimate the value of the Gini index before calculating it. For example, if a node is close to a 1:1 ratio of two classes, the Gini index is close to 0.5.

The Gain of the Gini index is calculated by subtracting the weighted average of the child nodes' impurity from the parent node's impurity. The formula for calculating the Gain is:

$$
Gain = Gini(parent) - \sum_{\text{child} \in \text{children}} \frac{\text{the number of data in the child}}{\text{the number of data in the parent}} \times Gini(child)
$$

Still, the Gini index has a limitation that it tends to prefer splits that result in a large number of partitions even if the partitions are small. This causes the risk of overfitting and the complex model.

**3) Entropy**





















### References



### Practices
