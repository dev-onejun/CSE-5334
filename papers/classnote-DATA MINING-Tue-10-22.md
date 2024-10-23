$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

precision: among all the predicted positive, how many are actually positive.
recall: among all the actual positive, how many are predicted positive.

* Methods for Performance Evaluation

cf. binary classificaiton에서 30% accuracy모델은 flipped만 하면 70% accuracy와 같다

- Factors that affect the performance of a model
$1.$ Class distribution
$2.$ Misclassification costs
$3.$ Size of training and test sets
    + Holdout
    + .
        + Problem
        + The ratio of classes in the training and test sets may not be the same

    - Random subsampling
    - Repeated holdout: 각 iteration에서 새로운 모델을 만든다.  (평균을 내는건가?)
        - Problem
        - 각 샘플(데이터)들이 train, test에서 뽑히는 횟수가 달라, weight가 다르게 모델에 반영된다

    + Cross-validation
    + k-fold cross-validation: iterate K times, each time, 1/k of the data is used for testing, and the rest is used for training
    + The maximum value of k (partition) is n (Leave-one-out)
    + This is a good way to estimate the performance of a model, applying both easy and hard samples evenly.
        + Leave-one-out Cross Validation
        + Leave-one-out has no randomness, but it is computationally expensive.
    + k값은 보통(많은 실험경험적으로) 10으로 설정한다?
        + Repeated Stratified cross-validation

    - Stratified sampling
    -

    + Bootstrap
    + Sampling with replacement

    * A note for parameter tuning with test data results
    - It is important that the test data is not used in any circumstances for training the model. The chance of submitting the result is only once. (TEST DATA SHOULD BE USED ONLY ONCE)
    $\to$ therefore, to adjust hyperparameters with the training result, validation data is used.

    * After performing evaluation during training step, validation data is used for hyperparameter tuning (training?) but isn't it make the model change?: traiditonal 모델들은 train 한번하면 끝나니까. deep learning은 epoch가 여러번 돌자너.

* Methods for Model Comparision

- ROC (Receiver Operating Characteristic)
- 각 axis는 어떤 사건에 대해 True Positive일 확률, False Positive일 확률을 나타낸다.
























### References

$\tag*{}\label{n} \text{[n] }$
