from sympy.algebras.quaternion import Quaternion as Q
from sympy import symbols,sin,cos,invert,solve,simplify,sqrt,I,log
from sympy import collect,expand,factor,trigsimp
from sympy.simplify.fu import TR7, TR5

x0,y0,z0=symbols('x0,y0,z0')
x1,y1,z1,x2,y2,z2,w1,w2=symbols('x1,y1,z1,x2,y2,z2,w1,w2')
e1,e2,e3=symbols('e1,e2,e3')
a=symbols('a')
X,Y,Z=symbols('X,Y,Z')

q=Q(cos(a/2),e1*sin(a/2),e2*sin(a/2),e3*sin(a/2))
def conjugate_quaternion(quat):
    return Q(quat.a,-quat.b,-quat.c,-quat.d)
conj_q=conjugate_quaternion(q)

p1=Q(0.,x1,y1,z1)
p0=Q(0.,x0,y0,z0)

#из общая формулы p1=q*p0*q^-1 нужно найти угол
p1=(q.mul(p0)).mul(conj_q)

# res=solve(x1-p1.b,a)
eq1=p1.b-x1 #-e1*(-e1*x0*sin(a/2) - e2*y0*sin(a/2) - e3*z0*sin(a/2))*sin(a/2) + e2*(e1*y0*sin(a/2) - e2*x0*sin(a/2) + z0*cos(a/2))*sin(a/2) - e3*(-e1*z0*sin(a/2) + e3*x0*sin(a/2) + y0*cos(a/2))*sin(a/2) - x1 + (e2*z0*sin(a/2) - e3*y0*sin(a/2) + x0*cos(a/2))*cos(a/2)
eq1_simply=e1**2*x0*sin(a/2)**2 + 2*e1*e2*y0*sin(a/2)**2 + 2*e1*e3*z0*sin(a/2)**2 - e2**2*x0*sin(a/2)**2 + e2*z0*sin(a) - e3**2*x0*sin(a/2)**2 - e3*y0*sin(a) + x0*cos(a/2)**2 - x1
#ищем угол а solve(eq1_simply,a)
eq1_res=[-I*log((e1**2*x0 + 2*e1*e2*y0 + 2*e1*e3*z0 - e2**2*x0 - e3**2*x0 + x0 - 2*x1 - 2*sqrt(e1**2*x0**2 - e1**2*x0*x1 + 2*e1*e2*x0*y0 - 2*e1*e2*x1*y0 + 2*e1*e3*x0*z0 - 2*e1*e3*x1*z0 - e2**2*x0**2 + e2**2*x0*x1 - e2**2*z0**2 + 2*e2*e3*y0*z0 - e3**2*x0**2 + e3**2*x0*x1 - e3**2*y0**2 - x0*x1 + x1**2))/(e1**2*x0 + 2*e1*e2*y0 + 2*e1*e3*z0 - e2**2*x0 + 2*I*e2*z0 - e3**2*x0 - 2*I*e3*y0 - x0)), -I*log((e1**2*x0 + 2*e1*e2*y0 + 2*e1*e3*z0 - e2**2*x0 - e3**2*x0 + x0 - 2*x1 + 2*sqrt(e1**2*x0**2 - e1**2*x0*x1 + 2*e1*e2*x0*y0 - 2*e1*e2*x1*y0 + 2*e1*e3*x0*z0 - 2*e1*e3*x1*z0 - e2**2*x0**2 + e2**2*x0*x1 - e2**2*z0**2 + 2*e2*e3*y0*z0 - e3**2*x0**2 + e3**2*x0*x1 - e3**2*y0**2 - x0*x1 + x1**2))/(e1**2*x0 + 2*e1*e2*y0 + 2*e1*e3*z0 - e2**2*x0 + 2*I*e2*z0 - e3**2*x0 - 2*I*e3*y0 - x0))]

eq1_res1=-I*log((e1**2*x0 + 2*e1*e2*y0 + 2*e1*e3*z0 - e2**2*x0 - e3**2*x0 + x0 - 2*x1 - 2*sqrt(e1**2*x0**2 - e1**2*x0*x1 + 2*e1*e2*x0*y0 - 2*e1*e2*x1*y0 + 2*e1*e3*x0*z0 - 2*e1*e3*x1*z0 - e2**2*x0**2 + e2**2*x0*x1 - e2**2*z0**2 + 2*e2*e3*y0*z0 - e3**2*x0**2 + e3**2*x0*x1 - e3**2*y0**2 - x0*x1 + x1**2))/(e1**2*x0 + 2*e1*e2*y0 + 2*e1*e3*z0 - e2**2*x0 + 2*I*e2*z0 - e3**2*x0 - 2*I*e3*y0 - x0))
eq1_res2=-I*log((e1**2*x0 + 2*e1*e2*y0 + 2*e1*e3*z0 - e2**2*x0 - e3**2*x0 + x0 - 2*x1 + 2*sqrt(e1**2*x0**2 - e1**2*x0*x1 + 2*e1*e2*x0*y0 - 2*e1*e2*x1*y0 + 2*e1*e3*x0*z0 - 2*e1*e3*x1*z0 - e2**2*x0**2 + e2**2*x0*x1 - e2**2*z0**2 + 2*e2*e3*y0*z0 - e3**2*x0**2 + e3**2*x0*x1 - e3**2*y0**2 - x0*x1 + x1**2))/(e1**2*x0 + 2*e1*e2*y0 + 2*e1*e3*z0 - e2**2*x0 + 2*I*e2*z0 - e3**2*x0 - 2*I*e3*y0 - x0))

eq2=p1.c-y1
eq3=p1.d-z1

print("end")

#w
-e1*(-e2*z0*sin(a/2) + e3*y0*sin(a/2) - x0*cos(a/2))*sin(a/2) + e2*(-e1*z0*sin(a/2) + e3*x0*sin(a/2) + y0*cos(a/2))*sin(a/2) + e3*(e1*y0*sin(a/2) - e2*x0*sin(a/2) + z0*cos(a/2))*sin(a/2) + (-e1*x0*sin(a/2) - e2*y0*sin(a/2) - e3*z0*sin(a/2))*cos(a/2)
#i
-e1*(-e1*x0*sin(a/2) - e2*y0*sin(a/2) - e3*z0*sin(a/2))*sin(a/2) + e2*(e1*y0*sin(a/2) - e2*x0*sin(a/2) + z0*cos(a/2))*sin(a/2) - e3*(-e1*z0*sin(a/2) + e3*x0*sin(a/2) + y0*cos(a/2))*sin(a/2) + (e2*z0*sin(a/2) - e3*y0*sin(a/2) + x0*cos(a/2))*cos(a/2)
#j
-e1*(e1*y0*sin(a/2) - e2*x0*sin(a/2) + z0*cos(a/2))*sin(a/2) - e2*(-e1*x0*sin(a/2) - e2*y0*sin(a/2) - e3*z0*sin(a/2))*sin(a/2) - e3*(-e2*z0*sin(a/2) + e3*y0*sin(a/2) - x0*cos(a/2))*sin(a/2) + (-e1*z0*sin(a/2) + e3*x0*sin(a/2) + y0*cos(a/2))*cos(a/2)
#k
e1*(-e1*z0*sin(a/2) + e3*x0*sin(a/2) + y0*cos(a/2))*sin(a/2) - e2*(e2*z0*sin(a/2) - e3*y0*sin(a/2) + x0*cos(a/2))*sin(a/2) - e3*(-e1*x0*sin(a/2) - e2*y0*sin(a/2) - e3*z0*sin(a/2))*sin(a/2) + (e1*y0*sin(a/2) - e2*x0*sin(a/2) + z0*cos(a/2))*cos(a/2)


#p1 из прямой кинематики:
#i
expr_i = -e1*(-e1*x0*sin(a/2) - e2*y0*sin(a/2) - e3*z0*sin(a/2))*sin(a/2) + e2*(e1*y0*sin(a/2) - e2*x0*sin(a/2) + z0*cos(a/2))*sin(a/2) - e3*(-e1*z0*sin(a/2) + e3*x0*sin(a/2) + y0*cos(a/2))*sin(a/2) + (e2*z0*sin(a/2) - e3*y0*sin(a/2) + x0*cos(a/2))*cos(a/2) - X
expr_i2=-X + e1**2*x0/2 + e1*e2*y0 + e1*e3*z0 - e2**2*x0/2 - e3**2*x0/2 + x0/2 + (e2*z0 - e3*y0)*sin(a) + (-e1**2*x0/2 - e1*e2*y0 - e1*e3*z0 + e2**2*x0/2 + e3**2*x0/2 + x0/2)*cos(a)


# раскрываем скобки  expand
# e1**2*x0*sin(a/2)**2 + 2*e1*e2*y0*sin(a/2)**2 + 2*e1*e3*z0*sin(a/2)**2 - e2**2*x0*sin(a/2)**2 + 2*e2*z0*sin(a/2)*cos(a/2) - e3**2*x0*sin(a/2)**2 - 2*e3*y0*sin(a/2)*cos(a/2) + x0*cos(a/2)**2
# выносим за скобки sin и cos
# (e1**2*x0 + 2*e1*e2*y0 + 2*e1*e3*z0 - e2**2*x0 - e3**2*x0)*sin(a/2)**2 + x0*cos(a/2)**2 + 2*e2*z0*sin(a/2)*cos(a/2) - 2*e3*y0*sin(a/2)*cos(a/2)
# преобразуем к одному углу
# (e1**2*x0 + 2*e1*e2*y0 + 2*e1*e3*z0 - e2**2*x0 - e3**2*x0)*sin(a/2)**2 + x0*cos(a/2)**2 + (e2*z0 - e3*y0)*sin(a)
# преобразуем к одному углу
# (e1**2*x0 + 2*e1*e2*y0 + 2*e1*e3*z0 - e2**2*x0 - e3**2*x0)*(1-cos(a))/2 + x0*(1+cos(a))/2 + (e2*z0 - e3*y0)*sin(a)
# группируем вынося за скобки sin и cos
# e1**2*x0/2 + e1*e2*y0 + e1*e3*z0 - e2**2*x0/2 + e2*z0*sin(a) - e3**2*x0/2 - e3*y0*sin(a) + x0/2 + (-e1**2*x0/2 - e1*e2*y0 - e1*e3*z0 + e2**2*x0/2 + e3**2*x0/2 + x0/2)*cos(a)
# e1**2*x0/2 + e1*e2*y0 + e1*e3*z0 - e2**2*x0/2 - e3**2*x0/2 + x0/2 + (e2*z0 - e3*y0)*sin(a) + (-e1**2*x0/2 - e1*e2*y0 - e1*e3*z0 + e2**2*x0/2 + e3**2*x0/2 + x0/2)*cos(a) == X
# (e2*z0 - e3*y0)*sin(a) + (-e1**2*x0/2 - e1*e2*y0 - e1*e3*z0 + e2**2*x0/2 + e3**2*x0/2 + x0/2)*cos(a) == -(e1**2*x0/2 + e1*e2*y0 + e1*e3*z0 - e2**2*x0/2 - e3**2*x0/2 + x0/2) + X
# solve(a) - ниже два корня решения, решать нужно с учетом комплексных чисел
# [-2*atan((-e2*z0 + e3*y0 + sqrt(-X**2 + X*e1**2*x0 + 2*X*e1*e2*y0 + 2*X*e1*e3*z0 - X*e2**2*x0 - X*e3**2*x0 + X*x0 - e1**2*x0**2 - 2*e1*e2*x0*y0 - 2*e1*e3*x0*z0 + e2**2*x0**2 + e2**2*z0**2 - 2*e2*e3*y0*z0 + e3**2*x0**2 + e3**2*y0**2))/(X - e1**2*x0 - 2*e1*e2*y0 - 2*e1*e3*z0 + e2**2*x0 + e3**2*x0)),
#  2*atan((e2*z0 - e3*y0 + sqrt(-X**2 + X*e1**2*x0 + 2*X*e1*e2*y0 + 2*X*e1*e3*z0 - X*e2**2*x0 - X*e3**2*x0 + X*x0 - e1**2*x0**2 - 2*e1*e2*x0*y0 - 2*e1*e3*x0*z0 + e2**2*x0**2 + e2**2*z0**2 - 2*e2*e3*y0*z0 + e3**2*x0**2 + e3**2*y0**2))/(X - e1**2*x0 - 2*e1*e2*y0 - 2*e1*e3*z0 + e2**2*x0 + e3**2*x0))]


#j
# expr_j =  -e1*(e1*y0*sin(a/2) - e2*x0*sin(a/2) + z0*cos(a/2))*sin(a/2) - e2*(-e1*x0*sin(a/2) - e2*y0*sin(a/2) - e3*z0*sin(a/2))*sin(a/2) - e3*(-e2*z0*sin(a/2) + e3*y0*sin(a/2) - x0*cos(a/2))*sin(a/2) + (-e1*z0*sin(a/2) + e3*x0*sin(a/2) + y0*cos(a/2))*cos(a/2)
# collect(collect(expand(TR7(TR5(collect(trigsimp(expand(expr_j)),sin(a/2)**2)))),sin(a)),cos(a)) - универсальная функция для упрощения исходного выражения
# expr_j2 = -0.5*e1**2*y0 + 1.0*e1*e2*x0 + 0.5*e2**2*y0 + 1.0*e2*e3*z0 - 0.5*e3**2*y0 + 0.5*y0 + (-e1*z0 + e3*x0)*sin(a) + (e1**2*y0/2 - e1*e2*x0 - e2**2*y0/2 - e2*e3*z0 + e3**2*y0/2 + y0/2)*cos(a) -Y
# solve(expr_j2, a) - ниже два корня решения, решать нужно с учетом комплексных чисел
# [2.0*atan((-e1*z0 + e3*x0 + sqrt(-Y**2 - Y*e1**2*y0 + 2*Y*e1*e2*x0 + Y*e2**2*y0 + 2*Y*e2*e3*z0 - Y*e3**2*y0 + Y*y0 + e1**2*y0**2 + e1**2*z0**2 - 2*e1*e2*x0*y0 - 2*e1*e3*x0*z0 - e2**2*y0**2 - 2*e2*e3*y0*z0 + e3**2*x0**2 + e3**2*y0**2))/(Y + e1**2*y0 - 2.0*e1*e2*x0 - e2**2*y0 - 2.0*e2*e3*z0 + e3**2*y0)),
#  -2.0*atan((e1*z0 - e3*x0 + sqrt(-Y**2 - Y*e1**2*y0 + 2*Y*e1*e2*x0 + Y*e2**2*y0 + 2*Y*e2*e3*z0 - Y*e3**2*y0 + Y*y0 + e1**2*y0**2 + e1**2*z0**2 - 2*e1*e2*x0*y0 - 2*e1*e3*x0*z0 - e2**2*y0**2 - 2*e2*e3*y0*z0 + e3**2*x0**2 + e3**2*y0**2))/(Y + e1**2*y0 - 2.0*e1*e2*x0 - e2**2*y0 - 2.0*e2*e3*z0 + e3**2*y0))]



#k
# expr_k = e1*(-e1*z0*sin(a/2) + e3*x0*sin(a/2) + y0*cos(a/2))*sin(a/2) - e2*(e2*z0*sin(a/2) - e3*y0*sin(a/2) + x0*cos(a/2))*sin(a/2) - e3*(-e1*x0*sin(a/2) - e2*y0*sin(a/2) - e3*z0*sin(a/2))*sin(a/2) + (e1*y0*sin(a/2) - e2*x0*sin(a/2) + z0*cos(a/2))*cos(a/2)
# expr_k2= -Z - e1**2*z0/2 + e1*e3*x0 - e2**2*z0/2 + e2*e3*y0 + e3**2*z0/2 + z0/2 + (e1*y0 - e2*x0)*sin(a) + (e1**2*z0/2 - e1*e3*x0 + e2**2*z0/2 - e2*e3*y0 - e3**2*z0/2 + z0/2)*cos(a)
# solve(expr_k2, a)
# [-2*atan((-e1*y0 + e2*x0 + sqrt(-Z**2 - Z*e1**2*z0 + 2*Z*e1*e3*x0 - Z*e2**2*z0 + 2*Z*e2*e3*y0 + Z*e3**2*z0 + Z*z0 + e1**2*y0**2 + e1**2*z0**2 - 2*e1*e2*x0*y0 - 2*e1*e3*x0*z0 + e2**2*x0**2 + e2**2*z0**2 - 2*e2*e3*y0*z0 - e3**2*z0**2))/(Z + e1**2*z0 - 2*e1*e3*x0 + e2**2*z0 - 2*e2*e3*y0 - e3**2*z0)),
#  2*atan((e1*y0 - e2*x0 + sqrt(-Z**2 - Z*e1**2*z0 + 2*Z*e1*e3*x0 - Z*e2**2*z0 + 2*Z*e2*e3*y0 + Z*e3**2*z0 + Z*z0 + e1**2*y0**2 + e1**2*z0**2 - 2*e1*e2*x0*y0 - 2*e1*e3*x0*z0 + e2**2*x0**2 + e2**2*z0**2 - 2*e2*e3*y0*z0 - e3**2*z0**2))/(Z + e1**2*z0 - 2*e1*e3*x0 + e2**2*z0 - 2*e2*e3*y0 - e3**2*z0))]
