$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

#### Support Vector Machine

- Find a linear hyperplane (decision boundary) that separates the data, and the found hyperplane has the maximum margin to 최적의 threshold를 찾기 위해서.
- If the data is n-dimensional, the hyperplane is n-1 dimensional.

- 모든 클래스의 어떤 데이터에 대해서 margin이 최대가 되는 hyperplane을 찾는다.

- Mathmatically, the hyperplane is defined as $w^Tx + b = 0$ where $w$ is the normal vector to the hyperplane, $x$ is the data point, and $b$ is the bias term.
    - With the hyperplane above,
        The data point $x$ is classified as $1$ if $w^Tx + b \ge 1$ and $-1$ if $w^Tx + b \le -1$.

            cf.
            $$
            \frac{x}{a} + \frac{y}{b} = 1
            $$
            를 만족하는 직선은 $(a, 0)$, $(0, b)$를 지나는 직선이다.
            ex. (3,0), (0,2)를 지나는 직선은 $\frac{x}{3} + \frac{y}{2} = 1$이다.

    - The margin is defined as $\frac{2}{||w||}$ where $||w||$ is the norm of the normal vector $w$.

    - The goal is to maximize the margin $\frac{2}{||w||}$ subject to the constraints $y_i(w^Tx_i + b) \ge 1$ for all $i$ and $y_i(w^Tx_i + b) \le -1$ for all $i$.

    - $n-1$ hyperplane 을 찾기 위해서, 최소 $n+1$개의 데이터가 필요하다.
            - 식을 보면, $w^Tx + b = 0$이므로, $w$의 차원은 $x$의 차원과 같다.
            - b까지 +1 개의 값을 찾아야 하므로, 최소 n+1개의 데이터가 필요하다.

























### References

$\tag*{}\label{n} \text{[n] }$
