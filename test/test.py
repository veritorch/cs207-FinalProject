import veritorch.veritorch as vt
import numpy as np
import math
import pytest

def test_get_diff_scalar_to_scalar():
    sol=vt.Solver(1)
    def f(x):
        return [x*x]
    dx=sol.get_diff(f,[1])
    assert (dx==np.array([2])).all()

def test_get_diff_vector_to_scalar():
    sol=vt.Solver(2)
    def f(x,y):
        return [x*y]
    dx=sol.get_diff(f,[1,2])
    assert (dx==np.array([2,1])).all()

def test_get_diff_scalar_to_vector():
    sol=vt.Solver(1)
    def f(x):
        return [x*x*x, 4*x]
    dx=sol.get_diff(f,[2])
    assert (dx==np.array([[12],[4]])).all()

def test_get_diff_vector_to_vector():
    sol=vt.Solver(2)
    def f(x,y):
        return [x*y, x+y]
    dx=sol.get_diff(f,[1,2])
    assert (dx==np.array([[2,1],[1,1]])).all()

def test_get_diff_continuous_usage():
    sol=vt.Solver(2)
    def f(x,y):
        return [x*y]
    def g(x,y):
        return [x**3, 4*x]
    dx=sol.get_diff(f,[1,2])
    assert (dx==np.array([2,1])).all()
    dx=sol.get_diff(g,[2,1])
    assert (dx==np.array([[12,0],[4,0]])).all()

#test backward
def testb_get_diff_scalar_to_scalar():
    sol=vt.Solver(1)
    def f(x):
        return [x*x]
    dx=sol.get_diff(f,[1],mode="backward")
    assert (dx==np.array([2])).all()

def testb_get_diff_vector_to_scalar():
    sol=vt.Solver(2)
    def f(x,y):
        return [x*y]
    dx=sol.get_diff(f,[1,2],mode="backward")
    assert (dx==np.array([2,1])).all()

def testb_get_diff_scalar_to_vector():
    sol=vt.Solver(1)
    def f(x):
        return [x*x*x, 4*x]
    dx=sol.get_diff(f,[2],mode="backward")
    assert (dx==np.array([[12],[4]])).all()

def testb_get_diff_vector_to_vector():
    sol=vt.Solver(2)
    def f(x,y):
        return [x*y, x+y]
    dx=sol.get_diff(f,[1,2],mode="backward")
    assert (dx==np.array([[2,1],[1,1]])).all()

def testb_get_diff_continuous_usage():
    sol=vt.Solver(2)
    def f(x,y):
        return [x*y]
    def g(x,y):
        return [x**3, 4*x]
    dx=sol.get_diff(f,[1,2],mode="backward")
    assert (dx==np.array([2,1])).all()
    dx=sol.get_diff(g,[2,1],mode="backward")
    assert (dx==np.array([[12,0],[4,0]])).all()

def test_neg():
    sol=vt.Solver(3)
    x1=sol.create_variable(4)
    assert (-x1).x == -4, "error with neg"
    assert ((-x1).dx == np.array([-1,0,0])).all(), "error with neg"

def test_add():
    sol=vt.Solver(3)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    x3=sol.create_variable(6)
    f = x1+x2+x3
    assert f.x == 15, "error with add"
    assert (f.dx == np.array([1,1,1])).all(), "error with add"

def test_radd():
    sol=vt.Solver(3)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    x3=sol.create_variable(6)
    f = 4+x2+x3
    assert f.x == 15, "error with radd"
    assert (f.dx == np.array([0,1,1])).all(), "error with radd"

def test_sub():
    sol=vt.Solver(3)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    x3=sol.create_variable(6)
    f = x1-x2-3
    assert f.x == -4, "error with sub"
    assert (f.dx == np.array([1,-1,0])).all(), "error with sub"

def test_rsub():
    sol=vt.Solver(3)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    x3=sol.create_variable(6)
    f = 11-x1-x3
    assert f.x == 1, "error with rsub"
    assert (f.dx == np.array([-1,0,-1])).all(), "error with rsub"

def test_mul():
    sol=vt.Solver(3)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    x3=sol.create_variable(6)
    f = x1*x2*x3*2
    assert f.x == 240, "error with mul"
    assert (f.dx == np.array([60., 48., 40.])).all(), "error with mul"

def test_rmul():
    sol=vt.Solver(3)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    x3=sol.create_variable(6)
    f = 2*x1*x2*x3
    assert f.x == 240, "error with rmul"
    assert (f.dx == np.array([60., 48., 40.])).all(), "error with rmul"

def test_truediv():
    sol=vt.Solver(3)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    x3=sol.create_variable(6)
    f = x2/x1
    f2 = x2/2
    f3 = x2/(2*x1)
    f4 = x2/(x1*x1)
    assert f.x == 1.25, "error with truediv"
    assert (f.dx == np.array([-0.3125,  0.25  ,  0.    ])).all(), "error with truediv"
    assert f2.x == 2.5, "error with truediv"
    assert (f2.dx == np.array([0. , 0.5, 0. ])).all(), "error with truediv"
    assert f3.x == 0.625, "error with truediv"
    assert (f3.dx == np.array([-0.15625,  0.125  ,  0.])).all(), "error with truediv"
    assert f4.x == 0.3125, "error with truediv"
    assert (f4.dx == np.array([-0.15625,  0.0625 ,  0.])).all(), "error with truediv"

def test_rtruediv():
    sol=vt.Solver(3)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    x3=sol.create_variable(6)
    f = 1/x1
    f2 = 1/(2*x1)
    f3 = 7/(x1*x1)
    assert f.x == 0.25, "error with rtruediv"
    assert (f.dx == np.array([-0.0625, -0.    , -0.    ])).all(), "error with rtruediv"
    assert f2.x == 0.125, "error with rtruediv"
    assert (f2.dx == np.array([-0.03125, -0.     , -0.   ])).all(), "error with rtruediv"
    assert f3.x == 0.4375, "error with rtruediv"
    assert (f3.dx == np.array([-0.21875, -0.     , -0.     ])).all(), "error with rtruediv"

def test_pow():
    sol=vt.Solver(2)
    x1=sol.create_variable(4)
    x2=sol.create_variable(5)
    f = (x1+x2) ** 2
    assert f.x == 81, "error with pow"
    assert (f.dx == np.array([18., 18.])).all(), "error with pow"

def test_exp():
    sol=vt.Solver(2)
    x1=sol.create_variable(0)
    x2=sol.create_variable(5)
    f = np.exp(x1) + x2
    assert f.x == 6.0, "error with exp"
    assert (f.dx == np.array([1., 1.])).all(), "error with exp"

def test_log():
    sol=vt.Solver(2)
    x1=sol.create_variable(10)
    x2=sol.create_variable(5)
    f = np.log(x1) + np.log(x2)
    assert f.x == 3.9120230054281464, "error with log"
    assert (f.dx == np.array([0.1, 0.2])).all(), "error with log"

def test_sin():
    sol=vt.Solver(2)
    x1=sol.create_variable(math.pi/2)
    x2=sol.create_variable(math.pi/6)
    f = np.sin(x1) + np.sin(x2)
    assert f.x == 1.5, "error with sin"
    assert ((f.dx - np.array([6.12323400e-17, 8.66025404e-01])) < 10**(-8)).sum() == 2, "error with sin"

def test_cos():
    sol=vt.Solver(2)
    x1=sol.create_variable(math.pi/2)
    x2=sol.create_variable(math.pi/6)
    f = np.cos(x1) + np.cos(x2)
    assert f.x == 0.8660254037844388, "error with cos"
    assert ((f.dx - np.array([-1.,-0.5])) < 10**(-8)).sum() == 2, "error with cos"

def test_tan():
    sol=vt.Solver(2)
    x1=sol.create_variable(math.pi/2)
    x2=sol.create_variable(math.pi/6)
    f = np.tan(x1) + np.tan(x2)
    assert f.x == 1.633123935319537e+16, "error with tan"
    assert ((f.dx - np.array([2.66709379e+32, 1.33333333e+00])) < 10**(-8)).sum() == 2, "error with tan"

def test_arcsin():
    sol=vt.Solver(2)
    x1=sol.create_variable(0.5)
    x2=sol.create_variable(0.1)
    f = np.arcsin(x1) + np.arcsin(x2)
    assert abs(f.x - 0.6237661967598587)<1e-8, "error with arcsin"
    assert ((f.dx - np.array([1.15470054, 1.00503782])) < 10**(-8)).sum() == 2, "error with arcsin"

def test_arccos():
    sol=vt.Solver(2)
    x1=sol.create_variable(0.5)
    x2=sol.create_variable(0.1)
    f = np.arccos(x1) + np.arccos(x2)
    assert abs(f.x - 2.5178264568299342)<1e-8, "error with arccos"
    assert ((f.dx - np.array([-1.15470054, -1.00503782])) < 10**(-8)).sum() == 2, "error with arccos"

def test_arctan():
    sol=vt.Solver(2)
    x1=sol.create_variable(0.5)
    x2=sol.create_variable(0.1)
    f = np.arctan(x1) + np.arctan(x2)
    assert abs(f.x - 0.5633162614919682)<1e-8, "error with arctan"
    assert ((f.dx - np.array([0.8, 0.99009901])) < 10**(-8)).sum() == 2, "error with arctan"

def test_over_create():
    sol=vt.Solver(1)
    x1=sol.create_variable(1)
    with pytest.raises(Exception):
        x2=sol.create_variable(2)

def test_get_variable():
    sol=vt.Solver(2)
    x1=sol.create_variable(1)
    x1=sol.get_variable(0)

def test_str():
    sol=vt.Solver(2)
    x1=sol.create_variable(1)
    print(x1)

def test_divide_by_zero():
    sol=vt.Solver(2)
    x1=sol.create_variable(1)
    with pytest.raises(ValueError):
        x2=x1/0

def test_divide_by_zero_variable():
    sol=vt.Solver(2)
    x1=sol.create_variable(1)
    x2=sol.create_variable(0)
    with pytest.raises(ValueError):
        f=x1/x2

def test_rtruediv_by_zero():
    sol=vt.Solver(2)
    x1=sol.create_variable(0)
    with pytest.raises(ValueError):
        f=1.5/x1

def test_arcsin_out_of_range():
    sol=vt.Solver(2)
    x1=sol.create_variable(10)
    with pytest.raises(ValueError):
        f=np.arcsin(x1)

def test_arccos_out_of_range():
    sol=vt.Solver(2)
    x1=sol.create_variable(10)
    with pytest.raises(ValueError):
        f=np.arccos(x1)


#test backward mode variable
def testb_neg():
    x1=vt.Variable_b(4)
    a = -x1
    a.grad_value = 1.0
    assert a.value == -4, "error with neg"
    assert x1.grad() == -1, "error with neg"

def testb_add():
    x1=vt.Variable_b(4)
    x2=vt.Variable_b(5)
    x3=vt.Variable_b(6)
    f = x1+x2+x3
    f.grad_value = 1.0
    assert f.value == 15, "error with add"
    assert x1.grad() == 1, "error with add"
    assert x2.grad() == 1, "error with add"
    assert x3.grad() == 1, "error with add"

def testb_radd():
    x1=vt.Variable_b(4)
    x2=vt.Variable_b(5)
    x3=vt.Variable_b(6)
    f = 4+x2+x3
    f.grad_value = 1.0
    assert f.value == 15, "error with radd"
    assert x1.grad() == 0, "error with radd"
    assert x2.grad() == 1, "error with radd"
    assert x3.grad() == 1, "error with radd"

def testb_sub():
    x1=vt.Variable_b(4)
    x2=vt.Variable_b(5)
    x3=vt.Variable_b(6)
    f = x1-x2-3
    f.grad_value = 1.0
    assert f.value == -4, "error with sub"
    assert x1.grad() == 1, "error with sub"
    assert x2.grad() == -1, "error with sub"
    assert x3.grad() == 0, "error with sub"

def testb_rsub():
    x1=vt.Variable_b(4)
    x2=vt.Variable_b(5)
    x3=vt.Variable_b(6)
    f = 11-x1-x3
    f.grad_value = 1.0
    assert f.value == 1, "error with rsub"
    assert x1.grad() == -1, "error with rsub"
    assert x2.grad() == 0, "error with rsub"
    assert x3.grad() == -1, "error with rsub"

def testb_mul():
    x1=vt.Variable_b(4)
    x2=vt.Variable_b(5)
    x3=vt.Variable_b(6)
    f = x1*x2*x3*2
    f.grad_value = 1.0
    assert f.value == 240, "error with mul"
    assert x1.grad() == 60, "error with mul"
    assert x2.grad() == 48, "error with mul"
    assert x3.grad() == 40, "error with mul"

def testb_rmul():
    x1=vt.Variable_b(4)
    x2=vt.Variable_b(5)
    x3=vt.Variable_b(6)
    f = 2*x1*x2*x3
    f.grad_value = 1.0
    assert f.value == 240, "error with rmul"
    assert x1.grad() == 60, "error with rmul"
    assert x2.grad() == 48, "error with rmul"
    assert x3.grad() == 40, "error with rmul"

def testb_truediv():
    x1=vt.Variable_b(4)
    x2=vt.Variable_b(5)
    f = x2/x1
    f.grad_value = 1.0
    assert f.value == 1.25, "error with truediv"
    assert x1.grad() == -0.3125, "error with truediv"
    assert x2.grad() == 0.25, "error with truediv"

    x21=vt.Variable_b(4)
    x22=vt.Variable_b(5)
    f2 = x22/4
    f2.grad_value = 1.0
    assert f2.value == 1.25, "error with truediv"
    assert x21.grad() == 0, "error with truediv"
    assert x22.grad() == 0.25, "error with truediv"
    
    x31=vt.Variable_b(4)
    x32=vt.Variable_b(5)
    f3 = x32/(4*x31)
    f3.grad_value = 1.0
    assert f3.value == 5/16, "error with truediv"
    assert x31.grad() == -0.078125, "error with truediv"
    assert x32.grad() == 0.0625, "error with truediv"
    
    x41=vt.Variable_b(4)
    x42=vt.Variable_b(5)
    f4 = x42/(x41*x41)
    f4.grad_value = 1.0
    assert f4.value == 0.3125, "error with truediv"
    assert x41.grad() == -0.15625, "error with truediv"
    assert x42.grad() == 0.0625, "error with truediv"

def testb_rtruediv():
    x1=vt.Variable_b(4)
    x2=vt.Variable_b(5)
    f = 5/x1
    f.grad_value = 1.0
    assert f.value == 1.25, "error with rtruediv"
    assert x1.grad() == -0.3125, "error with rtruediv"
    assert x2.grad() == 0, "error with rtruediv"

    x21=vt.Variable_b(4)
    x22=vt.Variable_b(5)
    f2 = 4/(x21*x21)
    f2.grad_value = 1.0
    assert f2.value == 0.25, "error with rtruediv"
    assert x21.grad() == -0.125, "error with rtruediv"
    assert x22.grad() == 0, "error with rtruediv"

def testb_pow():
    x1=vt.Variable_b(4)
    x2=vt.Variable_b(5)
    f = (x1+x2) ** 2
    f.grad_value = 1.0
    assert f.value == 81, "error with pow"
    assert x1.grad() == 18, "error with pow"
    assert x2.grad() == 18, "error with pow"

def testb_exp():
    x1=vt.Variable_b(0)
    x2=vt.Variable_b(5)
    f = np.exp(x1) + x2
    f.grad_value = 1.0
    assert f.value == 6, "error with exp"
    assert x1.grad() == 1, "error with exp"
    assert x2.grad() == 1, "error with exp"

def testb_log():
    x1=vt.Variable_b(10)
    x2=vt.Variable_b(5)
    f = np.log(x1) + np.log(x2)
    f.grad_value = 1.0
    assert f.value == 3.9120230054281464, "error with exp"
    assert x1.grad() == 0.1, "error with exp"
    assert x2.grad() == 0.2, "error with exp"

def testb_sin():
    x1=vt.Variable_b(math.pi/2)
    x2=vt.Variable_b(math.pi/6)
    f = np.sin(x1) + np.sin(x2)
    f.grad_value = 1.0
    assert f.value == 1.5, "error with sin"
    assert abs(x1.grad() -6.12323400e-17) < 10**(-8), "error with sin"
    assert abs(x2.grad() -8.66025404e-01) < 10**(-8), "error with sin"

def testb_cos():
    x1=vt.Variable_b(math.pi/2)
    x2=vt.Variable_b(math.pi/6)
    f  = np.cos(x1) + np.cos(x2)
    f.grad_value = 1.0
    assert f.value == 0.8660254037844388, "error with cos"
    assert abs(x1.grad() - (-1)) < 10**(-8), "error with cos"
    assert abs(x2.grad() - (-0.5)) < 10**(-8), "error with cos"

def testb_tan():
    x1=vt.Variable_b(math.pi/2)
    x2=vt.Variable_b(math.pi/6)
    f = np.tan(x1) + np.tan(x2)
    f.grad_value = 1.0
    assert f.value == 1.633123935319537e+16, "error with tan"
    assert abs(x1.grad() - (2.667093788113571e+32)) < 10**(-8), "error with tan"
    assert abs(x2.grad() - (1.33333333e+00)) < 10**(-8), "error with tan"

def testb_arcsin():
    x1=vt.Variable_b(0.5)
    x2=vt.Variable_b(0.1)
    f = np.arcsin(x1) + np.arcsin(x2)
    f.grad_value = 1.0
    assert abs(f.value - 0.6237661967598587) < 1e-8, "error with arcsin"
    assert abs(x1.grad() - (1.15470054)) < 10**(-8), "error with arcsin"
    assert abs(x2.grad() - (1.00503782)) < 10**(-8), "error with arcsin"

def testb_arccos():
    x1=vt.Variable_b(0.5)
    x2=vt.Variable_b(0.1)
    f = np.arccos(x1) + np.arccos(x2)
    f.grad_value = 1.0
    assert abs(f.value - 2.5178264568299342) < 1e-8, "error with arccos"
    assert abs(x1.grad() - (-1.15470054)) < 10**(-8), "error with arccos"
    assert abs(x2.grad() - (-1.00503782)) < 10**(-8), "error with arccos"

def testb_arctan():
    x1=vt.Variable_b(0.5)
    x2=vt.Variable_b(0.1)
    f = np.arctan(x1) + np.arctan(x2)
    f.grad_value = 1.0
    assert abs(f.value - 0.5633162614919682) < 1e-8, "error with arctan"
    assert abs(x1.grad() - (0.8)) < 10**(-8), "error with arctan"
    assert abs(x2.grad() - (0.99009901)) < 10**(-8), "error with arctan"
