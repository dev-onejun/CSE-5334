$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

**강의자료 p.32까지만 이번 quiz.**

* final exam 준비 tip
    - cheat sheet에 다 쓰지 말고, main formula (ex. entropy, gini) 등만 정리해라
    - cheat sheet 1장? (Front and back). 몇 장이든 상관 없는듯

---
* 지난 시간 복습
    - Suppose 15 unique items.
        - how many possible itemsets?
            - $2^{15}$ or $2^{15} - 1$. both are correct
            $\to$ too many to compute
        - Hash tree
            - {1,2,3,5,6} 다음에 {2,3,5,6}인 이유는, {2,1,3,5,6}이 되면, 순서가 있다는 것이기 때문?
---

* Factors affecting complexity
    1. Minimum support threshold
    2. Dimensionality of data
    3. Size of data
    4. Average transaction width

- 이전까지는 frequent itemset generation에 대한 것이었음

* Rule Generation
    * Anti-monotone property
        - Confidence 식이 support와 관련이 있기 때문에,
            C(ABC $\to$ D)와 C(AB $\to$ D)는 관련이 없지만,
            C(ABC $\to$ D)와 C(AB $\to$ CD), C(A $\to$ BCD)는 관련이 있음
                - numerator가 같기 때문에. S(ABCD)

* Pattern Evaluation
    - want to measure whether the rule is interesting or not

    * Drawback of Confidence
        - although Tea $\to$ Coffee satisfy .75 confidence, it may not be interesting
            - because Coffee itself is very popular which 90% of people buy it

    * Lift
        - if the value is less than 1, it means that the rule is not interesting (negatively associated)
        - if the value is greater than 1, it means that the rule is interesting (positively associated)
        * However, Lift values can be overestimated or underestimated since it is a ratio calculation.
    * Interest
    * PS
    * $\phi$-coefficient

    - 그 외 여러 식들이 있으나, 위의 3개(lift, interest, phi) 식 정도만 기억할 것

    * Piatetsky-Shapiro (PS)

    ...

    * Property under variable permutations
        - 어떤 건 순서 바꾸면 ㅂ뀌고, 다른 건 안바뀜
    * Property under Inversion Operations

    * $\phi$-coefficient

    * Property under Null Addition
        - Null Addition: x y가 아닌 다른 transaction을 데이터에 추가하는 것
        - Invariant는 안바뀌고, 다른 것은 바뀜










### References

$\tag*{}\label{n} \text{[n] }$
