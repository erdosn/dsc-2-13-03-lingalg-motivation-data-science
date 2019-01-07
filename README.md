
# Motivation for Linear Algebra in Data Science

## Introduction

Algebra & Statistics are founding steps for data science & machine learning. Machine learning is as much about linear algebra, probability theory and statistics (especially graphical models) and information theory as much as data analysis. In this section of the course, we shall focus at Linear Algebra, which is a pre-requisite for almost any technical field of knowledge today in about any technical discipline, including computer science and data science.  This lesson attempts to present a quick motivation on how and why linear algebra is a valuable skills for data analysts.


## Objectives

You will be able to:

* Understand and state the importance of linear algebra in the fields of data science and machine learning
* Describe the areas in AI and machine learning where Linear Algebra might be used for advanced analytics


```python
import pandas as pd
import numpy as np
```

## Objectives
YWBAT
* Define linear algebra operations
    * dot product, cross products (potentially)
    * dot product - the sum of element wise multiplication of vectors
    * dot in python - 
        * vectors - dot product
        * matrices - matrix multiplication
* Define terms
    * identity matrix
        * diagonal matrix of 1s and 0s
    * left multiplication, right multiplication
        * that's a thing now
    * orthogonal
        * angle between vectors is 90 degrees
* Perform linear algebra operations in python using numpy
* How does LinAlg relate to Data Science?

### How does linear algebra relate to what we've been doing so far?
* linear regression can only be done comparing 2 things...
    * ex: compare sqft_living to price
    * ex: compare sqft_living, bedrooms to price
    * But how does this use linear algebra
        * 
* y = mx + b (algebra)
* y = m_x_ + b (linear algebra)

### What is linear algebra? Field of Mathematics
Contains elements
* scalars - numbers
* vectors - list of numbers of dimension n x 1
* matrices - 
    * list of numbers but multidimensional m x n
    * collection of vectors
    * define some space


### Where have we seen matrices?
* Dataframe is literally a matrix (pandas)


```python
# compare and contrast vectors and matrices in python
# how do we create a vector/identify a vector?

# is v a vector
v = np.array([1, 2, 3, 4])
vv = np.matrix(v)
print(v) # this is a vector
print(vv) # it is a matrix
```

    [1 2 3 4]
    [[1 2 3 4]]



```python
v.shape, vv.shape
```




    ((4,), (1, 4))




```python
x = np.array([3, 17])
y = np.array([31, 5])
```


```python
# actual dot product (Linear algebra)
# x dot y = 
3*31 + 17*5
```




    178



### Properties of Linear Algebra Dot Product
* Always returns a scalar (single number)
* non uniqueness (important later)
* communative/commutative /flippable

### Testing python's/numpy's .dot method


```python
np.dot(x, y) # acts like a normal dot product
```




    178




```python
x.dot(y) # acts like a normal dot product
```




    178




```python
np.dot(y, x) # acts like a normal dot product
```




    178



### Not really 'math' but still doing math stuff in python


```python
x = np.matrix([3, 17])
y = np.matrix([31, 5])
print(y, x)
print(x.shape, y.shape)
```

    [[31  5]] [[ 3 17]]
    (1, 2) (1, 2)



```python
# this fails because of matrix multiplication rules
# inner dimensions don't match

print(np.dot(x, y)) # <- broken because inner dimensions don't match
```


    -------------------------------------------------------------

    ValueError                  Traceback (most recent call last)

    <ipython-input-17-b31be6eedbe4> in <module>()
          2 # inner dimensions don't match
          3 
    ----> 4 print(np.dot(x, y)) # <- broken because inner dimensions don't match
    

    ValueError: shapes (1,2) and (1,2) not aligned: 2 (dim 1) != 1 (dim 0)



```python
# let's 'flip' y and make it a 2x1
print(y.T, y.T.shape)
```

    [[31]
     [ 5]] (2, 1)



```python
# returns a matrix (singleton)
print(x.shape, y.T.shape)
np.dot(x, y.T) # returns a 1x1 matrix
```

    (1, 2) (2, 1)





    matrix([[178]])




```python
x.dot(y.T) 
```




    matrix([[178]])




```python
# what happened here?
y.T.dot(x) # return a matrix with dimensions = outer dimensions (2x2)
```




    matrix([[ 93, 527],
            [ 15,  85]])




```python
# let's look into it
```


```python
print(y.T.shape, x.shape) 
```

    (2, 1) (1, 2)



```python
x*y.T # dot on matrices is same as multiplication
```




    matrix([[178]])




```python
y.T*x # dot on matrices is same as multiplication
```




    matrix([[ 93, 527],
            [ 15,  85]])




```python
print(y.T, x)
```

    [[31]
     [ 5]] [[ 3 17]]



```python
# interesting properties of dot products
# 2 vectors (v1, v2) are orthogonal iff the angle between v1 and v2 is perpendicular (90 degrees, pi/2 radians)
```


```python
v1 = np.array([1, 0])
v2 = np.array([0, 1])
```


```python
v1.dot(v2)
```




    0




```python
v2.dot(v1)
```




    0




```python
v1 = np.array([1, 1])
v2 = np.array([-1, 1])
v1.dot(v2)
```




    0




```python
# if 2 vectors are orthogonal their dot product is 0
# if dot product is 0 for 2 vectors then they are orthogonal
```

### Identity element - what does this do?
* additive identity, in arithmetic, is 0
    * a + 0 = a
* multiplicative identity, in arithmetic, is 1
    * a * 1 = a
* additive inverse of a number a, in arithmetic, is -a
    * a + (-a) = 0, adding returns the additive identity
    * Does every number have an additive inverse? yes, kind of, zero is it's own inverse (additive)
* multiplicative inverse of a number a, in arithmetic, is 1/a
    * a * (1/a) = 1, multiplying returns the multiplicative identity
    * Does every number have a mult. inverse? No!!!! Because 1/0 does not exist and 1 is its own inverse



### Identity elements in Linear Algebra
* Additive identity in LinAlg is a zero matrix of a distinct shape
    * A + [0] = A
* Multiplicative identity in LinAlg is a 1 matrix (identity matrix) and it's diagonal of shape nxn
    * AxI = A
* Additive inverse of A is -A
* Multiplicative inverse of A is A_inv 
    * A x A_inv = I
    * A has to have a square shape
    * ex: A is 2x3 2x3 * 2x3 -> impossible
        * What do we do?
        * left inverse and a right inverse
        * left inverse
            * (3x2) x (2x3)
            * A_inv_left * A = I (3x3)
        * right inverse
            * (2x3) x (3x2)
            * A     x A_inv_right = I (2x2)


```python
print(y, -y)
```

    [[31  5]] [[-31  -5]]



```python
# how to make the additive identity in python
np.zeros((2, 3))
```




    array([[0., 0., 0.],
           [0., 0., 0.]])




```python
# how to make the identity matrix in python
np.ones((2, 3))
```




    array([[1., 1., 1.],
           [1., 1., 1.]])




```python
np.identity(4)
```




    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])




```python
A = np.array([[1, 2],[3, 4]])
A
```




    array([[1, 2],
           [3, 4]])




```python
A_inv = np.linalg.inv(A)
A_inv
```




    array([[-2. ,  1. ],
           [ 1.5, -0.5]])




```python
A.dot(A_inv)
```




    array([[1.00000000e+00, 1.11022302e-16],
           [0.00000000e+00, 1.00000000e+00]])




```python
A*A_inv
```




    array([[-2. ,  2. ],
           [ 4.5, -2. ]])



## Background 

Linear Algebra is a continuous form of mathematics and is applied throughout science and engineering because it allows you to model natural phenomena and to compute them efficiently. Because it is a form of continuous and not discrete mathematics, a lot of computer scientists don’t have a lot of experience with it. Linear Algebra is also central to almost all areas of mathematics like geometry and functional analysis. Its concepts are a crucial prerequisite for understanding the theory behind Data Science. You don’t need to understand Linear Algebra before getting started in Data Science, but at some point, you may want to gain a better understanding of how the different Machine Learning algorithms really work under the hood. So if you really want to be a professional in this field, you will have to master the parts of Linear Algebra that are important for Machine Learning.



<img src="linalg.jpg" width=600>

You might already know a number of linear algebraic concepts and operations like matrix multiplication, calculating determinants, cross-products and eigenvectors/eigenvalues etc (dont worry if you dont, we shall cover these in the course). As a data scientist, it is imperative that you carry a theoretic, as well as a practical understanding of these and other similar concepts alongside their applications in real world problem solving. 

### An Analogy 

Think of a simple example where you first learn about a `sine` function as an infinite polynomial while learning trigonometry. Students normally practice this function by passing different values to it and getting the expected results, and manage to relate this to triangles and vertices. When learning advanced physics, students get to learn more applications of sine and other similar functions in the area of sound and light. In the domain of Signal Processing for 1D data, these functions pop up again to help you solve filtering, time-series related problems. An introduction to numeric computation around sine functions can not alone help you understand its wider application areas. In fact sine functions are everywhere in the universe from music to light/sound/radio waves, from pendulum oscillations to alternating current. 


## Why Linear Algebra?

*Linear algebra is the branch of mathematics concerning vector spaces and linear relationships between such spaces. It includes the study of lines, planes, and subspace, but is also concerned with properties common to all vector spaces.*

Analogous to the example we saw above, It is important that a data scientist understands how data structures are built with vectors and matrices following the geometric intuitions that are from linear algebra, in addition to the numeric calculations. Such data-focused understanding of linear algebra is what let's machine learning practitioners decide what tools can be applied to a given problem, how to interpret the results of experiments, whereas a numeric understanding helps towards the applications of these tools. 

A good understanding of Linear Algebra is necessary to analyze ML/AI algorithms, especially for Deep Learning where so much happens behind the scenes.  

Following are some of the areas where linear algebra is commonly practiced in the domain of data science and machine learning. 

### Computer Vision / Image Processing

![](https://www.researchgate.net/profile/Dennis_Wee_Neo/publication/260038465/figure/fig1/AS:297304901865489@1447894420867/Results-of-image-processing-by-MATLAB-a-RGB-matrix-for-original-image-b-simu-lated.png)

So we know that computers are designed to process binary information only, i.e. only 0s and 1s. So how can an image such as Einstein's face above with multiple attributes like color be stored in a computer? This is achieved by storing the pixel intensities for red, blue and green colors in a matrix format. Color intensities can be coded into this matrix and can be processed further for analysis and enhancement related tasks. So any operation which we perform on this image would likely use Linear Algebra with matrices at the back end.

### Deep Learning - Tensors

Deep Learning is a sub-domain of machine learning, concerned with algorithms that can imitate the functions and structure of a biological brain as a computational algorithm. These are called the artificial neural networks (ANNs). 

The algorithms usually store and process data in form of mathematical entities called tensors. A tensor is often thought of as a generalized matrix. That is, it could be a 1-D matrix (a vector is actually such a tensor),a 2-D matrix (like a data frame), a 3-D matrix (something like a cube of numbers), even a 0-D matrix (a single number), or a higher dimensional structure that is harder to visualize.

![](http://www.big-data.tips/wp-content/uploads/2017/10/tensor-machine-learning-illustration.jpg)

As shown in the image above where different input features are being extracted and stored as spatial locations inside a tensor which appears as a cube. A tensor encapsulates the scalar, vector and the matrix characteristics. For deep learning, creating and processing tensors and operations that are performed on these also require knowledge of linear algebra. 

### Natural Language Processing

NLP is another active area in Machine Learning is dealing with text data.  The most common techniques employed in NLP include BoW (Bag of Words) representation, Term Document Matrix etc. As shown in the image below, words in documents 1,2 and 3 , and their relations are being encoded as numbers and stored in a matrix format.

![](http://cdn-images-1.medium.com/max/1760/1*svLRt3OwVyqZiyDammWqiA.png)

NLP techniques work in a very similar manner to store counts (or something similar) of words found in different documents. These counts are store in a Matrix form to perform tasks like Semantic analysis, Language translation, Language generation etc.

### Dimensionality Reduction

Dimensionality reduction techniques which are heavily employed in the field of big data, use matrices to process data in order to reduce its dimensions. Principle Component Analysis (PCA) is a widely used dimenionality reduction technique, relies solely on calculating Eigenvectors and Eigenvalues to identify principle components, as a set of highly reduced dimensions. The picture below is an example of a three dimensional data being mapped into two dimensions using matrix manipulations. 

![](http://www.nlpca.org/fig_pca_principal_component_analysis.png)

---

#### Ok , that makes sense. So what is involved  ?

We can have some ideas on whats involved in field of linear algebra by having a quick glance at the word-cloud below that attempts to highlight key linear algebraic terms.

<img src="http://faculty.tru.ca/rtaylor/math1300/linalg_wordcloud.jpg" width = 500>


We'll go through an introductory series of lessons and labs that will cover basic ideas of linear algebra: an understanding of vectors and matrices with some basic operations that can be performed on these mathematical entities. We shall implement these ideas in python, in an attempt to give you foundation knowledge towards dealing with these algebraic entities and their properties. These skills will be applied in advanced machine learning sections later in the course. 

## Further Reading 

[Youtube: Why Linear Algebra](https://github.com/learn-co-curriculum/dsc-2-13-03-lingalg-motivation)

[Boost your data science skills. Learn linear algebra.](https://towardsdatascience.com/boost-your-data-sciences-skills-learn-linear-algebra-2c30fdd008cf)

[Quora: Applications of Linear Algebra in Deep Learning](https://www.quora.com/What-are-the-applications-of-linear-algebra-in-machine-learning)

## Summary 

In this lesson, we worked towards developing a motivation for learning linear algebra for data analaysis and machine learning. We looked at some use cases in practical machine learning problems where linear algebra and matrix manipulation might come in handy. In the following lessons, we shall look at some of these manipulations, working our way towards solving a regression problem using linear algebraic operations only. 
