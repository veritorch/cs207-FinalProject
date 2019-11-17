import numpy as np
# Solver class
# This class takes as an input the number of independent variables that the function  𝑓  has and tracks how many independent variables we have already created so far.

# It has the following attributes:

# n: the number of independent variables the function  𝑓  has
# independent_variable_list: a list of independent variables that we have created so far using this solver class
# We plan to implement the following methods for this solver class:

# create_variable(x, dx=1): return an independent variable with value initialized to x and derivative initialized to dx. A copy of this independent variable will also be added to the independent_variable_list.
# get_variable(idx): return the copy of the  𝑖𝑡ℎ  independent variable stored in the independent_variable_list
# merge(*args): *args should be a list of variables  [𝑓1,𝑓2,...,𝑓𝑚] . This function returns the m by n jacobian matrix of  𝑓=[𝑓1,𝑓2,...,𝑓𝑚] .

class Solver():
  
  def __init__(self, n):
    self.n = n 
    # NOTE (see more in NOTE under def get_variable): what's the use of this? 
    # We find this to be not very useful (other than using len in create_variable, 
    # in which case better to just use a counter variable rather than a list that takes more memory)
    self.independent_variable_list = []   
    
  def create_variable(self, x):  
    # NOTE: we deleted "dx=1" - PLEASE UPDATE DOCUMENTATION, 
    # since presumably create_variable only takes x as a number, and user never know the derivative at x (otherwise why use our package?)
    
    # check num of variables created is <= n, if so, append to the list; else raise exception
      
    der = np.zeros(self.n)
    der[len(self.independent_variable_list)] = 1
    var = Variable(x, der)
    
    if len(self.independent_variable_list) < self.n: 
      self.independent_variable_list.append(var)
    else:
      raise Exception ("Trying to create more variables than specified") 
      
    return var
  
  def get_variable(self, idx):  
    # NOTE: again, don't see the point this method (from both user's and dev's perspectives). 
    # For user: they may not remember the sequence of create_variable, which is the sequence in independent_variable_list.
    # even if they do, why would they use this method?
    # For dev: don't see where this could be used. Shuying mentioned that you might use this in merge method.
    # if this is the only use, please make this private. But I don't see how merge method needs this. 
    # You can always retrieve value of a variable by calling var.x (or get method in Variable class). This avoids index confusion
    # aka the need to remember user input sequence.
    stored_var = self.independent_variable_list[idx]
    return Variable(stored_var.x, stored_var.dx)
  
  def merge(self, *args):
    # we don't need to implement this for a scalar function
    pass 
  
  def __repr__(self):
    pass
  
  def __str__(self):
    pass

  
# This class takes as inputs the initial value  𝑥  and derivative  𝑑𝑥  (optional, set to 1 by default) and includes methods to overload basic arithmic operators for the veritorch package.

# It has the following attributes:

# x: the current scalar value of this variable
# dx: the current derivative of this variable, which should be a vector of length n, where n is the number of independent variables of the function  𝑓  whose derivative is being evaluated.
class Variable():
  # User can ONLY create instance of Variable class through Solver.create_variable method.
  # NOT to be used directly - IS THIS TRUE?
  def __init__(self, x, dx = None):
    """ 
    Initiate Variable object.
    
    Parameters
    =======
    x: int/flot; point of evaluation
    dx: np.array; with the position of partial derivative set to 1
    
    Returns
    =======
    None
  
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(4)
    >>> print(x1)
    Variable(4, [1. 0.])
    """
    self.x = x
    self.dx = dx
  
  def __repr__(self):
    # TODO
    pass
  
  def __str__(self):
    """ 
    Returns string representation of Variable object.
    
    Parameters
    =======
    Variable object (self)
    
    Returns
    =======
    string "Variable(" + str(self.x) + ", " + str(self.dx) + ")"
  
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(4)
    >>> str(x1)
    Variable(4, [1. 0.])
    """
    return "Variable(" + str(self.x) + ", " + str(self.dx) + ")"
  
  def __neg__(self):
    """ 
    Returns negative of Variable object.
    
    Parameters
    =======
    Variable object (self)
    
    Returns
    =======
    Variable object: -self
  
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(4)
    >>> v = -x1 
    >>> print(v)
    Variable(-4, [-1. -0.])
    """
    return Variable(-self.x, -self.dx)
  
  def __add__(self, other):
    """ 
    Returns addition of Variable object.
    
    Parameters
    =======
    Variable object (self)
    Variable object (other) OR float/int (other)
    
    Returns
    =======
    Variable object: self + other
  
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(4)
    >>> x2 = sol.create_variable(5)
    >>> v = x1 + x2
    >>> print(v)
    Variable(9, [1. 1.])
    """
    try:
        val = self.x + other.x
        der = self.dx + other.dx
    except AttributeError:
        val = self.x + other
        der = self.dx
    return Variable(val, der)
  
  def __radd__(self, other):
    """ 
    Returns addition of Variable object.
    
    Parameters
    =======
    Variable object (self)
    float/int (other)
    
    Returns
    =======
    Variable object: other + self
  
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(4)
    >>> v = 3 + x1
    >>> print(v)
    Variable(7, [1. 0.])
    """
    return self.__add__(other)
  
  def __sub__(self, other):
    """ 
    Returns subtraction of Variable object.
    
    Parameters
    =======
    Variable object (self)
    Variable object (other) OR float/int (other)
    
    Returns
    =======
    Variable object: self - other
  
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(4)
    >>> x2 = sol.create_variable(5)
    >>> v = x1 - x2
    >>> print(v)
    Variable(-1, [ 1. -1.])
    """
    return self.__add__(-other)
  
  def __rsub__(self, other):
    """ 
    Returns subtraction of Variable object.
    
    Parameters
    =======
    Variable object (self)
    float/int (other)
    
    Returns
    =======
    Variable object: other - self
  
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(4)
    >>> v = 3 - x1
    >>> print(v)
    Variable(-1, [-1. -0.])
    """
    return -self.__sub__(other)
  
  def __mul__(self, other):
    """ 
    Returns multiplication of Variable object.
    
    Parameters
    =======
    Variable object (self)
    Variable object (other) OR float/int (other)
    
    Returns
    =======
    Variable object: self * other
  
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(4)
    >>> x2 = sol.create_variable(5)
    >>> v = x1 * x2
    >>> print(v)
    Variable(20, [5. 4.])
    """
    try: 
      val = other.x * self.x
      der = other.x * self.dx + other.dx * self.x 
    except AttributeError:
      val = other * self.x
      der = other * self.dx
    return Variable(val, der)
  
  def __rmul__(self, other):
    """ 
    Returns multiplication of Variable object.
    
    Parameters
    =======
    Variable object (self)
    float/int (other)
    
    Returns
    =======
    Variable object: other * self
  
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(4)
    >>> v = 5 * x
    >>> print(v)
    Variable(20, [5. 0.])
    """
    return self.__mul__(other)
  
  def __truediv__(self, other):
    """ 
    Returns division of Variable object.
    
    Parameters
    =======
    Variable object (self)
    Variable object (other) OR float/int (other)
    
    Returns
    =======
    Variable object: self/other
  
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(4)
    >>> x2 = sol.create_variable(5)
    >>> v = x1/x2
    >>> print(v)
    Variable(0.8, [ 0.2  -0.16])
    """
    if isinstance(other, int):
      if other == 0:
        raise ValueError("Cannot divide by 0")
    elif other.x ==0:
      raise ValueError("Cannot divide by 0")
    try: 
      val = np.divide(self.x, other.x)
      der = self.dx * (1/other.x) - self.x * (other.x) **(-2) * other.dx
    except AttributeError:
      val = np.divide(self.x, other)
      der = self.dx * (1/other)
    return Variable(val, der)
  
  def __rtruediv__(self, other):
    """ 
    Returns division of int/float and Variable object.
    
    Parameters
    =======
    Variable object (self)
    int/float (other)
    
    Returns
    =======
    Variable object: other/self
    
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(4)
    >>> v = 1/x1
    >>> print(v)
    Variable(0.25, [-0.0625 -0.    ])
    """
    if self.x == 0:
      raise ValueError("Cannot divide by 0")
    val = other/self.x
    der = -other * self.x ** (-2) * self.dx
    return Variable(val, der)
  
  def __pow__(self, p):
    """ 
    Returns the power of Variable object.
    
    Parameters
    =======
    Variable object (self)
    
    Returns
    =======
    Variable object with
    - x attribute is updated based on: x^p
    - dx attribute is updated based on chain rule: (x^p)' = p * x^(p-1) * x'
    
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(5)
    >>> v = x1 ** 3
    >>> print(v)
    Variable(125, [75.  0.])
    """
    val = self.x ** p
    der = p * self.x ** (p-1) * self.dx
    return Variable(val, der)
  
  def exp(self):
    """ 
    Returns the exponential of Variable object.
    
    Parameters
    =======
    Variable object (self)
    
    Returns
    =======
    Variable object with
    - x attribute is updated based on: exp(x)
    - dx attribute is updated based on chain rule: exp'(x) = exp(x) * x'
    
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(5)
    >>> v = np.exp(x1)
    >>> print(v)
    Variable(148.4131591025766, [148.4131591   0.       ])
    """
    val = np.exp(self.x)
    der = np.exp(self.x) * self.dx
    return Variable(val, der)
  
  def log(self):
    """ 
    Returns the natural log of Variable object.
    
    Parameters
    =======
    Variable object (self)
    
    Returns
    =======
    Variable object with
    - x attribute is updated based on: log(x)
    - dx attribute is updated based on chain rule: log'(x) = 1/x * x'
    
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(5)
    >>> v = np.log(x1)
    >>> print(v)
    Variable(1.6094379124341003, [0.2 0. ])
    """
    val = np.log(self.x)
    der = (1/self.x) * self.dx
    return Variable(val, der)
  
  def sin(self):
    # NOTE: implementing elementary function within Variable class, for good style.
    # PLEASE UPDATE documentation
    """ 
    Returns the sine of Variable object.
    
    Parameters
    =======
    Variable object (self)
    
    Returns
    =======
    Variable object with
    - x attribute is updated based on: sin(x)
    - dx attribute is updated based on chain rule: dx = cos(x) * dx
    
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> import math
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(math.pi/3)
    >>> v = np.sin(x1)
    >>> print(v)
    Variable(0.8660254037844386, [0.5 0. ])
    """
    val = np.sin(self.x)
    der = np.cos(self.x) * self.dx
    return Variable(val, der)
  
  def cos(self):
    """ 
    Returns the cosine of Variable object.
    
    Parameters
    =======
    Variable object (self)
    
    Returns
    =======
    Variable object with
    - x attribute is updated based on: cos(x)
    - dx attribute is updated based on chain rule: cos'(x) = -sin(x) * x'
    
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> import math
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(math.pi/3)
    >>> v = np.cos(x1)
    >>> print(v)
    Variable(0.5000000000000001, [-0.8660254 -0.       ])
    """
    val = np.cos(self.x)
    der = -np.sin(self.x) * self.dx
    return Variable(val, der)
  
  def tan(self):
    """ 
    Returns the tangent of Variable object.
    
    Parameters
    =======
    Variable object (self)
    
    Returns
    =======
    Variable object with
    - x attribute is updated based on: tan(x)
    - dx attribute is updated based on chain rule: 1/(cos(x))^2 * x'
    
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> import math
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(math.pi/4)
    >>> v = np.tan(x1)
    >>> print(v)
    Variable(0.9999999999999999, [2. 0.])
    """
    val = np.tan(self.x)
    der = 1/(np.cos(self.x))**2 * self.dx
    return Variable(val, der)
  
  def arcsin(self):
    """ 
    Returns the arcsine of Variable object.
    
    Parameters
    =======
    Variable object (self)
    
    Returns
    =======
    Variable object with
    - x attribute is updated based on: arcsin(x)
    - dx attribute is updated based on chain rule: 1/sqrt(1-x^2) * x'
    
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(0)
    >>> v = np.arcsin(x1)
    >>> print(v)
    Variable(0.0, [1. 0.])
    """
    if self.x >= 1 or self.x <= -1:
      raise ValueError("arcsin does not exist beyond (-1,1)")
    val = np.arcsin(self.x)
    der = 1/np.sqrt(1-self.x**2) * self.dx
    return Variable(val, der)
  
  def arccos(self):
    """ 
    Returns the arccosine of Variable object.
    
    Parameters
    =======
    Variable object (self)
    
    Returns
    =======
    Variable object with
    - x attribute is updated based on: arccos(x)
    - dx attribute is updated based on chain rule: -1/sqrt(1-x^2) * x'
    
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(0)
    >>> v = np.arcsin(x1)
    >>> print(v)
    Variable(1.5707963267948966, [-1. -0.])
    """
    if self.x >= 1 or self.x <= -1:
      raise ValueError("arccos does not exist beyond (-1,1)")
    val = np.arccos(self.x)
    der = -1/np.sqrt(1-self.x**2) * self.dx
    return Variable(val, der)
  
  def arctan(self):
    """ 
    Returns the arctangent of Variable object.
    
    Parameters
    =======
    Variable object (self)
    
    Returns
    =======
    Variable object with
    - x attribute is updated based on: arctan(x)
    - dx attribute is updated based on chain rule: 1/(1+x^2) * x'
    
    Examples
    =======
    >>> import numpy as np
    >>> import veritorch as vt
    >>> sol = vt.Solver(2)
    >>> x1 = sol.create_variable(0)
    >>> v = np.arctan(x1)
    >>> print(v)
    Variable(0.0, [1. 0.])
    """
    val = np.arctan(self.x)
    der = 1/(1+self.x**2) * self.dx
    return Variable(val, der)
