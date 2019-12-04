import numpy as np
from inspect import signature
class Solver():
# Solver class
# This class takes as an input the number of independent variables that the function  𝑓  has and tracks how many independent variables we have already created so far.

# It has the following attributes:

# n: the number of independent variables the function  𝑓  has
# independent_variable_list: a list of independent variables that we have created so far using this solver class
# We plan to implement the following methods for this solver class:

# create_variable(x, dx=1): return an independent variable with value initialized to x and derivative initialized to dx. A copy of this independent variable will also be added to the independent_variable_list.
# get_variable(idx): return the copy of the  𝑖𝑡ℎ  independent variable stored in the independent_variable_list
# merge(*args): *args should be a list of variables  [𝑓1,𝑓2,...,𝑓𝑚] . This function returns the m by n jacobian matrix of  𝑓=[𝑓1,𝑓2,...,𝑓𝑚] .
    def __init__(self, n):
        self.n = n
        self.independent_variable_list = []
    
    #create variable for forward mode
    def create_variable(self, x):
        der = np.zeros(self.n)
        der[len(self.independent_variable_list)] = 1
        var = Variable(x, der)
        if len(self.independent_variable_list) < self.n: 
            self.independent_variable_list.append(var)
        else:
            raise Exception ("Trying to create more variables than specified") 

        return var
    
    #create variable for backward mode
    def create_variable_b(self, x):
        var_b = Variable_b(x)
        if len(self.independent_variable_list) < self.n:
            self.independent_variable_list.append(var_b)
        else:
            raise Exception ("Trying to create more variables than specified")
        return var_b
    
    def get_variable(self, idx):
        stored_var = self.independent_variable_list[idx]
        return Variable(stored_var.x, stored_var.dx)

    def merge(self, d_list):
    	return np.stack(d_list, axis=0)
        
    def get_diff_forward(self, f, x):
        #check arguments of f
        #f must take in n independent variables
        sig = signature(f)
        if(len(sig.parameters)!=self.n):
        	raise TypeError("# of arguments of f != n")
        #check length of x
        #user must provide n point values
        if(len(x)!=self.n):
            raise IndexError('len(x) != n')
        #reset solver
        self.independent_variable_list = []
        for i in range(self.n):
            self.create_variable(x[i])
        ans=f(*self.independent_variable_list)
        if(ans is None or len(ans)<=0):
            raise TypeError("# of outputs of f <= 0")
        elif(len(ans)>1):
            dans=[]
            for i in range(len(ans)):
                dans.append(ans[i].dx)
            #reset solver again
            self.independent_variable_list=[]
            return self.merge(dans)
        else:
            #reset solver again
            self.independent_variable_list=[]
            return ans[0].dx
    
    def get_diff_backward(self, f, x):
        #check arguments of f
        #f must take in n independent variables
        sig = signature(f)
        if(len(sig.parameters)!=self.n):
            raise TypeError("# of arguments of f != n")
        #check length of x
        #user must provide n point values
        if(len(x)!=self.n):
            raise IndexError('len(x) != n')
        
        #for backward mode, if the function has multiple outputs, due to our design issue, we must recreate all variable for each output of the function
        #evaluate f to get number of outputs first
        ans=f(*x)
        if(len(ans)<=0):
            raise TypeError("# of outputs of f <= 0")
        elif(len(ans)==1):
            self.independent_variable_list = []
            for i in range(self.n):
                self.create_variable_b(x[i])
            ans=f(*self.independent_variable_list)
            ans[0].grad_value = 1.0
            dx=[]
            for i in range(len(self.independent_variable_list)):
                dx.append(self.independent_variable_list[i].grad())
            dx=np.array(dx)
            self.independent_variable_list=[]
            return dx
        else:
            dans=[]
            for i in range(len(ans)):
                self.independent_variable_list=[]
                for j in range(self.n):
                    self.create_variable_b(x[j])
                ans=f(*self.independent_variable_list)
                ans[i].grad_value = 1.0
                dx=[]
                for j in range(len(self.independent_variable_list)):
                    dx.append(self.independent_variable_list[j].grad())
                dx=np.array(dx)
                self.independent_variable_list=[]
                dans.append(dx)
            return self.merge(dans)
            
    def get_diff(self, f, x, mode="forward"):
        if(mode=="forward"):
            return self.get_diff_forward(f,x)
        else:
            return self.get_diff_backward(f,x)

    def __repr__(self):
        pass
    
    def __str__(self):
        pass

    
class Variable():
# This class takes as inputs the initial value  𝑥  and derivative  𝑑𝑥  (optional, set to 1 by default) and includes methods to overload basic arithmic operators for the veritorch package.

# It has the following attributes:

# x: the current scalar value of this variable
# dx: the current derivative of this variable, which should be a vector of length n, where n is the number of independent variables of the function  𝑓  whose derivative is being evaluated.

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
        >>> v = np.arccos(x1)
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
    
    def __eq__(self, other):
        """
        Return true if two Variable objects are equal(both value and derivative must match).

        Parameters
        =======
        Variable object (self)
        Variable object (other) OR float/int (other)

        Returns
        =======
        true: if they are equal
        false: if they are not equal

        Examples
        =======
        >>> import numpy as np
        >>> import veritorch as vt
        >>> sol = vt.Solver(2)
        >>> x1 = sol.create_variable(4)
        >>> x2 = sol.create_variable(5)
        >>> v = x1 == x2
        >>> print(v)
        false
        """
        try:
            return (self.x == other.x) and ((self.dx == other.dx).all())
        except AttributeError:
            return False

    def __ne__(self, other):
        """
        Return true if two Variable objects are not equal.

        Parameters
        =======
        Variable object (self)
        Variable object (other) OR float/int (other)

        Returns
        =======
        true: if they are not equal
        false: if they are equal

        Examples
        =======
        >>> import numpy as np
        >>> import veritorch as vt
        >>> sol = vt.Solver(2)
        >>> x1 = sol.create_variable(4)
        >>> x2 = sol.create_variable(5)
        >>> v = x1 == x2
        >>> print(v)
        true
        """
        return not self.__eq__(other)

class Variable_b():
    
    def __init__(self, value):
        """
        Initiate Variable object.
    
        Parameters
        =======
        value: int/float; point of evaluation
        
        Returns
        =======
        None
        """
    
        self.value = value
        self.children = []
        self.grad_value = None #None means it is not yet evaluated
    
    def __repr__(self):
        #TODO
        pass

    def __str__(self):
        #TODO
        if self.grad_value == None:
            return "Variable_b(" + str(self.value) + ", " + "None" + ")"
        return "Variable_b(" + str(self.value) + ", " + str(self.grad_value) + ")"

    def grad(self):
        #TODO: DOCTEST
        if self.grad_value is None: # recurse only if the value is not yet cached
            self.grad_value = sum(weight * var.grad()
                    for weight, var in self.children)#chain rule to calc derivative
        return self.grad_value
    
    def __neg__(self):
        z = Variable_b(-self.value)
        self.children.append((-1, z))  # weight = ∂z/∂self = -1
        return z
    
    def __add__(self, other):
        try:
            z = Variable_b(self.value + other.value)
            other.children.append((1, z)) # weight = ∂z/∂self = 1
            self.children.append((1, z)) # weight = ∂z/∂other = 1
        except AttributeError:
            z = Variable_b(self.value + other)
            self.children.append((1, z)) # weight = ∂z/∂self = 1
        return z
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self.__add__(-other)
    
    def __rsub__(self, other):
        return -self.__sub__(other)
    
    def __mul__(self, other):
        try:
            z = Variable_b(self.value * other.value)
            self.children.append((other.value, z)) # weight = ∂z/∂self = other.value
            other.children.append((self.value, z)) # weight = ∂z/∂other = self.value
        except AttributeError:
            z = Variable_b(self.value * other)
            self.children.append((other, z)) # weight = ∂z/∂self = other.value
        return z
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, int):
            if other == 0:
                raise ValueError("Cannot divide by 0")
        elif other.value ==0:
            raise ValueError("Cannot divide by 0")
        try:
            z = Variable_b(self.value / other.value)
            self.children.append(((1/other.value), z)) # weight = ∂z/∂self = 1/other.value
            other.children.append((- self.value * (other.value) **(-2) , z)) # weight = ∂z/∂other = - self.value * (other.value) **(-2)
        except AttributeError:
            z = Variable_b(self.value / other)
            self.children.append(((1/other), z)) # weight = ∂z/∂self = 1/other.value
        return z
    
    def __rtruediv__(self, other):
        if self.value == 0:
            raise ValueError("Cannot divide by 0")
        z = Variable_b(other/self.value)
        self.children.append((-other * self.value ** (-2), z)) # weight = ∂z/∂elf = - other * (self.value) **(-2)
        return z
    
    def __pow__(self, p):
        z = Variable_b(self.value**p)
        self.children.append((p*self.value**(p-1), z)) # weight = ∂z/∂self = p*self.value**(p-1)
        return z
    
    def exp(self):
        z = Variable_b(np.exp(self.value))
        self.children.append((np.exp(self.value), z)) # weight = ∂z/∂self = exp(self.value)
        return z
    
    def log(self):
        z = Variable_b(np.log(self.value))
        self.children.append((1/self.value, z)) # weight = ∂z/∂self = 1/self.value
        return z
    
    def sin(self):
        z = Variable_b(np.sin(self.value))
        self.children.append((np.cos(self.value), z)) # weight = ∂z/∂self = np.cos(self.value)
        return z
    
    def cos(self):
        z = Variable_b(np.cos(self.value))
        self.children.append((-np.sin(self.value), z)) # weight = ∂z/∂self = -np.sin(self.value)
        return z
    
    def tan(self):
        z = Variable_b(np.tan(self.value))
        self.children.append((1/(np.cos(self.value))**2, z))
        return z
    
    def arcsin(self):
        if self.value >= 1 or self.value <= -1:
            raise ValueError("arcsin does not exist beyond (-1,1)")
        z = Variable_b(np.arcsin(self.value))
        self.children.append((1/np.sqrt(1-self.value**2), z))
        return z
    
    def arccos(self):
        if self.value >= 1 or self.value <= -1:
            raise ValueError("arccos does not exist beyond (-1,1)")
        z = Variable_b(np.arccos(self.value))
        self.children.append((-1/np.sqrt(1-self.value**2), z))
        return z
    
    def arctan(self):
        z = Variable_b(np.arctan(self.value))
        self.children.append((1/(1+self.value**2), z))
        return z
