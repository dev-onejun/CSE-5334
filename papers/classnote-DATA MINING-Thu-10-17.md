$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

* Limitation of accuracy metrics
- 80% C1 과 20% C2가 있다고 할 때, C1으로 모두 predict해도 80% accuracy가 나옴
- TP, TN, FP, FN을 반영하지 못한다

    - Cost matrix
    - TP, TN, FP, FN에 가중치를 부여해 최종 cost를 계산한다. 그러면 accuracy가 높더라도 cost가 높은 애들을 거를 수 있다.

    - Precision, Recall, F1 score
        - Recall is more important when the data is about 조금의 실수도 용납 안되는 Terrorist classification, Medical Diagnosis (반대의 것인 FP이 많아도 상관없는 경우를 상상해도 될듯?)
        - Precision is more important in (FN이 많아도 상관없는 경우) google search

### References

$\tag*{}\label{n} \text{[n] }$
