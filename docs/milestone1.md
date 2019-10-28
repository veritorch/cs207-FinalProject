
# <center>CS 207 Final Project: Milestone 1</center>
<center>October 2019</center>

## Introduction


Efficiently and accurately evaluating derivatives of functions is one of the most important operations in science and engineering. Automatic differentiation (AD) is a technique which, given a function $f$ and a point, automatically evaluates that point's derivative. AD is less costly than symbolic defferentiation, while achieving machine precision compared with finite differentiation. This library implements the forward mode of AD, along with some additional features.


## Background

We present below some of the key concepts and formulae upon which we build the veritorch library:

### Chain rule

Chain rule is fundamental to AD when we decompose functions.

Suppose we have $h(u(t), v(t))$, its derivative with respect to $t$ is:

$$\frac{\partial h}{\partial t} = \frac{\partial h}{\partial u}\frac{\partial u}{\partial t} + \frac{\partial h}{\partial v}\frac{\partial v}{\partial t}.$$

For the general function $h(y_1(\mathbf{x}), \dotsc,y_n(\mathbf{x}))$, where we replace $t$ with a vector $\mathbf{x} \in \mathbb{R}^m$ and $h$ a function of $n$ other functions $y_i$, the derivative is:

$$\nabla_x h = \sum_{i=1}^n \frac{\partial h}{\partial y_i} \nabla y_i(\mathbf{x})$$


### Computational graphs 

AD exploits the idea that complicated equations could be converted into a sequence of elementary operations which have specified routines for computing derivatives. This process, also called the evaluation trace, can be visualized by a computational graph where each step is an elementary operation. For example, we want to evaluate the derivative of the function: $$f(x) = 5\exp(x^2)+\sin(3x)$$
Here in this example, the right-most $x_7$ represents the value of $f(x)$, while the left-most $x$ represents our input variable. We construct a computational graph where we take the input $x$ as a node, and we take the constants as nodes as well when applicable. These nodes are connected by lines (edges) to represent the flow of information. 
![](https://i.imgur.com/uBUpnfc.jpg =400x)



### Elementary functions

Elementary Functions | Example | Derivative
:-:|:-:|:-:
exponentials| $e^x$ | $e^x$   
logarithms|$log(x)$| $\frac{1}{x}$ 
powers| $x^2$| $2x$ 
trigonometrics| $sin(x)$ | $cos(x)$ 
inverse trigonometrics|  $arcsin(x)$ | $\frac{1}{\sqrt{(1-x^2)}}$ 


### Dual number
$\forall z_1=x_1+y_1\epsilon, z_2=x_2+y_2\epsilon,$ where $x_1, y_1, x_2, y_2\in \mathbb{R}$, we have the following properties for dual number:
1. $z_1+z_2=(x_1+x_2)+(y_1+y_2)\epsilon$
2. $z_1z_2=(x_1x_2)+(x_1y_2+x_2y_1)\epsilon$
3. $z_1/z_2=(\frac{x_1}{x_2})+\frac{x_2y_1-x_1y_2}{x_2^2}$

As can be seen from the equations above, there is a close connection between the multiplication/division of dual numbers and the product/quotient rules for derivatives:
$~(f(x)g(x))'=f'(x)g(x)+f(x)g'(x),~(\frac{f(x)}{g(x)})'=\frac{f'(x)g(x)-f(x)g'(x)}{g^{2}(x)}$.


## How to Use PackageName

To use the veritorch package, users should first run the commands provided below to install our package via pip and import it. 

```
pip install veritorch
python
>>>import veritorch as vt
```

After successfully installing and importing the veritorch package, users can take the following steps to evaluate the derivative of $f$ at a point $x$. Here we take $f(x)=f(x_1,x_2,x_3)=x_1x_2x_3,~x=(x_1,x_2,x_3)=(4,5,6)$ as an example:

```
# First, create an instance of solver class in the veritorch package that
# tracks how many of independent variables $f$ takes as input.

>>>sol=vt.solver(3)

# Next, use the method create_variable(initial_value) of solver class
# to create variable x1, x2, x3 with their values initialized to 4, 5, 6 
# (and partial derivatives, with respect to x1, x2, x3 respectively 
# initialized to 1 by default)

>>>x1=sol.create_variable(4)
>>>x2=sol.create_variable(5)
>>>x3=sol.create_variable(6)
>>>f1=x1*x2*x3
>>>print(f1)
Value: 120, Derivative: [30,24,20]
```


The veritorch package will also support composite functions that involve elementary functions, including but not limited to $\sin(x), \cos(x), \exp(x), \arcsin(x)$:

```
# create variable x1 with its value initialized to pi 
# (and derivatives initialized to 1 by default)

>>>import math
>>>sol=vt.solver(1)
>>>x1=sol.create_variable(math.pi)
>>>f1=vt.sin(x1)
>>>f2=vt.cos(f1)
>>>f3=vt.exp(f2)
>>>f4=vt.arcsin(f3)

# user can also put in a composite function all at once. demo omitted.


# for multi-dimensional function, user can use solver.merge method 
# to get the jacobian matrix

>>>sol=vt.solver(2)
>>>x1=sol.create_variable(1)
>>>x2=sol.create_variable(2)
>>>f1=x1*x2
>>>f2=x1**x2
>>>f3=x1+2*x2
>>>print(sol.merge(f1, f2, f3))
Value: [1,1,5], Derivative: [[2,1],[2,0],[1,2]]
```
Then typing "print(f4)" will show us the final value and derivative of the composite function.


<!--
To instantiate the veritorch object, the user would be prompt to interact with shell and input the following, step by step:
1. An integer $n$ representing the number of variables 
2. An integer $m$ representing the number of functions
3. A list of funtions with $m$ functions, with variable names being $\{x_1, x_2, ..., x_n\}$
4. A list of numbers with $n$ values, at which the derivative of function(s) would be evaluated

Below is an example of shell prompt and user input:
```
>>> Input the number of variables you want
2
>>> Input the number of functions you want
3
>>> Please input a list of 3 functions, with variable names being x1, x2
[x1+x2, x1*x2, sin(x2)]
>>> Please input a list of 2 values, at which the functions would be evaluated
[2, 4]
```

`revision needed`
-->

## Software Organization

### Directory Structure

We plan to make the final repo follow the directory structure below:
```
cs207-FinalProject/
    README.md
    requirements.txt
    LICENSE
    setup.py
    veritorch/
        __init__.py
        veritorch.py
        utils.py
        ...
        test/
            test.py
            ...
    docs/
        milestone1.md
        demo.ipynb
        ...
```

### Modules

We expect our package to include the following modules:

1. NumPy: it provides us an efficient way to compute the intermediate multidimentional results and organize value and derivative vectors.

2. pytest: it provides us a systematic way to test every line of code written in the veritorch library

3. setuptools: used to package our repo before we distribute it.

### Testing
All files related to testing will be put in the directory cs207-FinalProject/veritorch/test/. We will use TravisCI to check whether each pull request can actually build and Codecov to check how many lines of code have been tested. We have activated both of them for this repo and included the badges in the README.md file at the root directory.

### Distrubution of the package
We will use twine to upload the veritorch package to PyPI (Python Packaging Index) for package distribution. It allows users to search and download packages by keywords or by filters using pip and pipenv.

## Implementation

### Class
There are two core classes in the veritorch package: the solver class and the variable class

#### Solver class

This class takes as an input the number of independent variables that the function $f$ has and tracks how many independent variables we have already created so far. 

It has the following attributes:
* n: the number of independent variables the function $f$ has
* independent_variable_list: a list of independent variables that we have created so far using this solver class

We plan to implement the following methods for this solver class:
* create_variable(x, dx=1): return an independent variable with value initialized to x and derivative initialized to dx. A copy of this independent variable will also be added to the independent_variable_list.
* get_variable(idx): return the copy of the $i^{th}$ independent variable stored in the independent_variable_list
* merge(*args): *args should be a list of variables $[f_1, f_2, ..., f_m]$. This function returns the m by n jacobian matrix of $f=[f_1, f_2, ..., f_m]$. 

#### Variable class
This class takes as inputs the initial value $x$ and derivative $dx$ (optional, set to 1 by default) and includes methods to overload basic arithmic operators for the veritorch package.

It has the following attributes:
* x: the current scalar value of this variable
* dx: the current derivative of this variable, which should be a vector of length n, where n is the number of independent variables of the function $f$ whose derivative is being evaluated.

We plan to implement the following methods for this variable class:
* \_\_add\_\_
* \_\_radd\_\_
* \_\_raddmul\_\_
* \_\_raddrmul\_\_
* \_\_raddtruediv\_\_
* \_\_raddrtruediv\_\_
* \_\_raddstr\_\_

### Elementary functions
To support the usage of elementary functions in our library, we will implement the following functions:
* exp(x)
* log(x)
* pow(x, y)
* sin(x)
* cos(x)
* tan(x)
* arcsin(x)
* arccos(x)
* arctan(x)

### External dependencies
NumPy, pytest, setuptools, TravisCI and Codecov.