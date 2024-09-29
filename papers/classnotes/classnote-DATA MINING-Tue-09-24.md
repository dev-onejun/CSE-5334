$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

GAIN = GINI(Parent) + GINI(Children) \
GINI(children) = $\sum_{\text{child} \in \text{children}} \frac{\text{the number of data in child}}{\text{the number of data in parent}} \times GINI(child)$

However, GINI has a limitation

**2. Alternative Splitting Criteria based on INFO - Entropy**

$$
\text{Entropy}(t) = - \sum_{j} p_{(j \mid t)} \log_2 p_{(j \mid t)}
$$
(log 식 분석했을 때, log 값은 항상 -이므로, 앞에 -를 붙여 entropy 값을 양수로 만듬)

- class의 비율이 같을 때 entropy값이 1이 됨
    - half-half일 때 1
    - 0:N일 때 0
- 갯수에 상관없이, 비율이 같으면 entropy값이 같음
    - 1:2, 2:1도 같은 entropy값을 가짐
- class 갯수에 상관 없이 maximum은 $\log_2 N_c$, where $N_c$ is the number of classes and minimum은 0

However, solely with Entropy, it tends to prefer splits that result in a large number of partitions even if the partitions are small.
('How to determine the Best Split' 페이지에서 student id로 나뉘는 것을 피하려는 거임)

**Split Info**

$$
Gain Ratio = \frac{\text{Gain}}{\text{Split Info}} \\
Split Info = - \sum_{\text{child} \in \text{children}} \frac{\text{the number of data in child}}{\text{the number of data in parent}} \times \log_2 \frac{\text{the number of data in child}}{\text{the number of data in parent}}
$$

Entropy랑 splitinfo랑 같이 proposed 되었다는 거 같은데

**3. Missclassificaiton error**

- simple 계산 팁
    It is the smallest number of data in the node.
    ex. 3:7 -> 3 => 3/10 / 5:5 -> 5 => 5/10
    ex. 3:3:3 => 3 => 6/9 (minority class가 3이고, 나머지 6은 majority가 된다고 봐야함)

$$
\text{Missclassification Error} = 1 - \max(p_{(j \mid t)})
$$

Comparing GINI, Entropy, and Missclassification Error, Entropy shows the most ? ("for a 2-class problem 강의자료 그래프")
    - for example in p.49, ME(P) = 3/10, ME(C1) = 0, ME(C2) = 3/7 => Gain = 0
                           Gini(P) = 0.42, Gini(C1) = 0, Gini(C2) = 0.489 => Gain = 0.08

    => GINI and Entropy is beter than Missclassification Error


**Stop Criteria**
Stop splitting when the node is pure or the number of data is less than the pre-defined threshold. 어느정도 내려가다보면 데이터 수가 적어지기 때문에, 그것들을 틀리는 게 accuracy (performance)에 큰 영향을 미치지 않는다. 오히려 overfitting이 될 수 있으니, 그것을 방지하기 위해 멈춘다.

**Advantages**
- Inexpensive to construct
- Extremely fast at classifying unknown records
- Easy to interpret the results why a decision was made (Explainable AI?)
- Accuracy is comparable to other classification techniques for many simple data sets, typically with a small dataset





### References

$\tag*{}\label{n} \text{[n] }$
