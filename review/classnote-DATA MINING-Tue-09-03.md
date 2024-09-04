$$
\begin{array}{|c|c|}
\hline
\text{} & \text{} \\
\hline
\text{} & \text{} \\
\hline
\end{array}
$$

* What is Data?
    - Data is a collection of objects and their attributes. Imagine a table with rows and columns. Each row is an object and each column is an attribute.

* What is Attribute values?
    - Attribute values are numbers, categories, or text. For example, in a table of students, the attributes could be student ID, name.
    - They are differnt with attributes. Same attributes can be mapped to different values. For example, height can be measured in inches or meters.
    - Same attribute values are able to be mapped to different attributes. For example, both student ID and age are integer, but they have different meanings.

Four types of attributes; 1) Nominal, 2) Ordinal, 3) Interval, 4) Ratio.

Discrete and Continuous attributes

The type of Attributes depends on the properties of the data. Four properties of data are **1)** Distinctness, **2)** Order, **3)** Addition, and **4)** Multiplication.

Normally, Distinctness is belong to all attribute

1) Nominal Attributes: distinctness
2) Ordinal attribute: distinctness, order
3) Interval attribute: distinctness, order, addition
4) Ratio attribute: distinctness, order, addition, multiplication
- ex. addtion이 없으면 multiplication 확인 필요 없음

Celcius cannot be applied as multiplication since 10C is not twice as hot as 5C. Kelvin?

Operations in Nominal, like mode, are enabled to be applied to Ordinal, Interval, and Ratio since they have the property of distinctness.

Important Characteristics of Structed Data
1) Dimensionality
2) Sparsity
3) Resolution

Dimensionality: The number of attributes in the data. The higher the dimensionality, the more complex the data is. The curse of dimensionality is a problem that occurs when the dimensionality of the data is too high. It is difficult to analyze and visualize data with high dimensionality.

Sparsity:

Resolution: The level of detail in the data.

Data Matrix only has a numerical data ?

Transaction Data: A collection of items that are bought together. Each row is a transaction and each column is an item. they are nominal attributes.

Graph Data: A collection of nodes and edges. Each row is a node and each column is an edge. They are nominal attributes.

Ordered Data: Different ffrom transaction data, ordered data is a sequences of transactions. ex. Genomic sequence data, Spatio-temporal data.

**Data Quality**

Noise is the unwanted data which modifies an desired data difficult to find.
Outliers are the data that are significantly different from the rest of the data.
Missing values are the data that are not available. They can be caused by errors in data collection or data processing. For example, . A few methods to handle missing values are that **1)** remove missed data object, **2)** estimate the missing values, **3)** ignore the missing values, and **4)** replace the missing values with all possible values like mean, median, and mode, weighted by available? probabilities.
Duplicate data are the data that are repeated in the data, majorly caused when mering data from heterogeneous sources.

**Data Preprocessing**

Data preprocesing is ..

Aggregation is the process of combining multiple data objects into a single object. The volume of data is reduced by aggregation so that the data becomes more stable, less variability.

Sampling is the process of selecting a subset of data objects from the original data. Normally, the entire data is too large and expensive to analyze regarding time and cost. Sampling is used to be adopted as a solution. While ~~, upsampling can be applied to increase the number of data objects.
Sampling should have same property with the original data.


### References

$\tag*{}\label{n} \text{[n] }$
