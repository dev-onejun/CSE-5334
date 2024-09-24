##### Classification: Decision Tree

Classification is to find a model that describes and distinguishes data classes or concepts, for the purpose of being able to use the model to predict the class of objects whose class label is unknown.

Train data is used to build the model. Validate data is 트레이닝 과정에서 unseen에 대해 검증하는 것. Test data는 주어지지 않은 데이터에 대해 검증하는 것

Classification vs. Prediction
Classification is for categorical data which is finite and discrete. It is most suited for nominal attributes but less effective for ordinal attributes. Prediction is for continuous data which is infinite and continuous. It is most suited for ordinal attributes.

Supervised vs. Unsupervised
Supervised learning is when the model is trained on a labeled dataset. Unsupervised learning is when the model is trained on an unlabeled dataset.
cf. Semi-supervised learning is when the model is trained on a partially labeled dataset which is the entire data is too big to label for cost perspective.

Decision Tree is a tree-like graph of decisions and their possible consequences. It has splitting attributes which makes a decision yes or no. Binary로 만들지, multi-way로 만들지는 사용자가 정할 수 있다.
If data has conflicting attributes, decision tree can not solve the problem.
Many decision trees can be made with the same data. Finding the optimal decision tree is NP-complete problem so that greedy algorithm is used to find the optimal tree; grow the tree by making locally optimal decisions in selecting the best attribute to split the data.

Optimal 선택 방법은 다음에 볼 것

Many algorithms for decision tree
One of them is Hunt's

Hunt's Algorithm is 1. pure?(label이 모두 다 같은가)-make as leaf node. if not-> 2. empty? 3. conflicting? 4. split based on the attribute
for the 3, all attributes should be conflicting?

YES:NO -> Refund에 대해 3:7 => 0:3 and 3:4. (3:4는 pure하지 않으므로 split)

Decision Tree Induction
- Large search space ?

How to specify the attribute test condition?
It depends on attribute types. For nominal attributes, Multi-way split or Binary split is used. For ordinal attributes, Multi-way split or Binary split is used - binary에서 (small, large) 가능.
For continuous attributes, ?
