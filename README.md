# CSE-5334-Assignments

## P1

### Related Files

``` plaintext
programming_assignment_1
├── __init__.py
├── p1.py
tests
├── __init__.py
├── test_p1.py
US_Inaugural_Addresses
├── 01_1789-Washington.txt
... 40 txt files
|
P1_guidelines.ipynb
```

### Tips

The tips were given at the class Sep. 19th, 2024. The original is written in the day of the classnote.

Programming Assignment 오래 걸릴거니까 빨리 시작해라 ,,
- TF-idf 계산식 직접 만들라는건가? 가져다 쓰지말고?
    - log-weighted tf and log-weighted idf are used
    - The TF-IDF vectors should be normalized, resulting in the length of each vector being 1.
        - TIP: 조금 해보고 맞는지 확인해라. -> 라이브러리로 확인하고 하면 될듯?
- The given guideline preprocessed 'lowercase', 'tokenization', 'stopword_removal', and 'stemming'. Those preprocessing steps need to apply in both query and document.
- ltc.lnc weighting scheme
- 6까지는 수업에서 한 거랑 같음
- 7 cosine sim calculate method
    - documents가 billion이면 다 계산 힘듬. 실제로 쓸 수 있는 방법이 있어야 함
        - posting list? **내용 맞는지는 확인해봐야함..**
            - 각 term에 대해 top k 리스트를 저장?
                - query가 들어왔을 때 그 term에 대한 posting list를 가져와서 계산
                - query에 2개 term이 있으면, in the worst case, k + k 개의 documents 계산. in the best K개.

                - k+k개에서 documents 종류가 다르면 query의 term에 대해 값을 가지고 있지 않을 수 있음
                    - top k로 sorting한 것이므로, 없는 term의 가장낮은 값으로 대체해 사용하면 됨. 그 값은 해당 term이 없는 문서가 가질 수 있는 최대값임(sorting한 것이므로)
                 
### Assessment Result

Got minus at IndexError at the `getidf()` function and results differences in TF-IDF values.

- Needed to use try-catch when accessing to the corpus
- ?
