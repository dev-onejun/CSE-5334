$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

* ROC Curve
    - TPR + FNR = 1
    - FPR + TNR = 1

    - 하나만 알면 다른 하나를 알 수 있기 때ㅐ문에, roc curve는 각 축을 TPR과 FPR로만 (각각 하나씩) 그린다

    - every threshold gives you a single point on the ROC Curve
        강의자료의 threshold가 움직이며 그때그때의 TPR, FPR값이 나오며, 그것을 그린 것이 ROC Curve다.

    - ROC Curve의 면적은 AUC (Area Under the Curve)로 표현한다.
        - AUC가 1에 가까울수록 좋은 모델이다.
        - AUC가 0.5에 가까울수록 좋지 않은 모델이다.

    - FPR 비율이 특정값보다 낮게 유지한다면 M1이 M2보다 좋은 모델이다.
    - 반대로 FPR 비율이 높게 나오는 상황이라면 M2가 M1보다 좋은 모델이다.

    - test instance들에 대해 확률값으로 예측을 뽑아놓으면, 각 확률을 threshold로 두고 roc curve를 그릴 수 있다.
        cf. 각 threshold에 대해 계산 안하고 1 / (# of positive sample 또는 # of negative sample )씩 (아래 또는 왼쪽으로) 옮겨가며 그릴 수도 있다

    시험에 무조건 나옴. 계산과정이 많으니 tip으로 계산하는 것 추천.
        - 책 practice에도 있으니 연습 꼭 해보기. 게산 안하는 방법으로.

    - 같은 확률로 있으면 -를 왼쪽으로, +를 오른쪽으로 table에 두고 그린다?






















### References

$\tag*{}\label{n} \text{[n] }$
