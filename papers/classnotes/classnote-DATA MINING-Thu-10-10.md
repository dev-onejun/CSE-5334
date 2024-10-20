$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

* Support Vector Machines
    - SVM is not changed if the data is added in the middle of each class. This is the difference from other models like Deicision tree, ...

    - What if the problem (data) is not linearly separable?
        - slack variables
            ?
    - not linear decision boundary data?
        - transform data into higher dimensional space, resulting in linearly separable data: kernel trick

* Artificial Neural Networks (ANN)
- a complex non-linear function

    - Perceptron
    $$
    y = \begin{cases} 1 & \text{if } w^Tx + b > 0 \\ -1 & \text{otherwise} \end{cases}
    $$
    - activation function: sign function (step function? not 미분)

        * Perceptron learning algorithm
        ``` plaintext
        Initialize the weights to 0 or small random numbers (not 0 in 실제로는. 계산량이 너무 많아지기 때문)
        For each training sample (x(i), y(i)):
            Compute the predicted output value (y_hat)
            Update the weights with
                w = w + learning_rate * (y(i) - y_hat) * x(i)
                (when it comes to error (=y(i) - y_hat),
                    if the error is 0, the weights are not updated
                    if the error is positive, the weights are increased
                    if the error is negative, the weights are decreased)
        ```

    However, the perceptron is not able to solve the XOR problem. Plus, the step function is not differentiable.
    (XOR Practice 직접 쌓아서 계산해보기)

    - Multi-layer Perceptron (MLP)

        Activation function에 lienar function 안쓰는 이유: non-linear 표현 불가

        problem: perceptron leraning algorithm 적용 불가
        -> backpropagation algorithm`with Gradient Descent

    - Design Issues in ANN ( hyperparameters?)
    -






























### References

$\tag*{}\label{n} \text{[n] }$
