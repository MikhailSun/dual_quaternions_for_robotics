import numpy as np
import makecharts as mc
import dual_quat_test as dqt
from dual_quat_test import DQ,link,frame
import matplotlib.pyplot as plt
from sympy import simplify, nsimplify, Derivative, diff,expand,sin,cos,Quaternion,symbols,solve

DH=dict(l1=dict(Tetta=0,d=0.507,a=0.265,alfa=90),)
        # l2=dict(Tetta=90,d=0,a=0.7,alfa=0),)
        # l3=dict(Tetta=0,d=0,a=0.02,alfa=90),
        # l4=dict(Tetta=0,d=0.759,a=0,alfa=90),
        # l5=dict(Tetta=0,d=0,a=0,alfa=-90),
        # l6=dict(Tetta=180,d=0.143,a=0,alfa=0),)

angles=list([10,-10,10,90,90,10])

links=[]
for link_name,DH_par in DH.items():
    if len(links)==0:
        origin_dq=None
    else:
        origin_dq = links[-1].origin1
    l=link(DH_par,origin_quat=origin_dq)
    links.append(l)

for a,l in zip(angles,links):
    l.transform(a)

#1) численное решение:
# cyfral=links[0].origin1.dq_to_matrix()
# print(cyfral)
#аналитическое решение (должно совпадать с численным если подставлено значение tetta):
# links[0].get_analytical(10)

#2) теперь пытаемся вывести в аналитическом виде матрицу вращения/перемещения ()
# analytical_dq0=links[0].get_analytical_dq()
# analytical_matrix0=links[0].get_matrix(smplfy=True)
# analytical_dq1=links[1].get_analytical_dq()
# analytical_matrix1=links[1].get_matrix(smplfy=True)
# print(analytical_dq0)
# print(analytical_dq1)
# print(analytical_matrix0)
# print(analytical_matrix1)

#3)         #из кватерниона нужно извлечь rx ry rz
        #попробуем решить так: попробуем решить так - умножим аналитически 3 кватерниона опследовательных поворотов относительно исходной СК X Y Z - это будем называть углами Эйлера (или самолетным )
        #1) расчет аналитического кватерниона на основе угла и оси
        #2) аналитическое умножение кватернионов
        #3) вывод из получившегося результата углв Эйлера

        # как перевести данные из кватерниона в rx ry rz
rx,ry,rz=symbols('rx,ry,rz')
RX=Quaternion.from_axis_angle(vector=(1.,0.,0.),angle=rx)
RY=Quaternion.from_axis_angle(vector=(0.,1.,0.),angle=ry)
RZ=Quaternion.from_axis_angle(vector=(0.,0.,1.),angle=rz)

QE=nsimplify(RX.mul(RY.mul(RZ)))

q0,q1,q2,q3=symbols('q0,q1,q2,q3')
eq0=q0-QE.a #q0 + sin(rx/2)*sin(ry/2)*sin(rz/2) - cos(rx/2)*cos(ry/2)*cos(rz/2)   (1) solve(eq0,sin(rx/2)*sin(ry/2))
eq1=q1-QE.b #q1 - sin(rx/2)*cos(ry/2)*cos(rz/2) - sin(ry/2)*sin(rz/2)*cos(rx/2)   (2)
eq2=q2-QE.c #q2 + sin(rx/2)*sin(rz/2)*cos(ry/2) - sin(ry/2)*cos(rx/2)*cos(rz/2)   (3)
eq3=q3-QE.d #q3 - sin(rx/2)*sin(ry/2)*cos(rz/2) - sin(rz/2)*cos(rx/2)*cos(ry/2)   (4) solve(eq3,sin(rx/2)*sin(ry/2))

#из (1) sin(rx/2)*sin(ry/2)=(-q0 + cos(rx/2)*cos(ry/2)*cos(rz/2))/sin(rz/2)
#из (4) sin(rx/2)*sin(ry/2)=(q3 - sin(rz/2)*cos(rx/2)*cos(ry/2))/cos(rz/2)
#приравниваем eq4=(-q0 + cos(rx/2)*cos(ry/2)*cos(rz/2))/sin(rz/2) - (q3 - sin(rz/2)*cos(rx/2)*cos(ry/2))/cos(rz/2)
#упрощаем: (-q0*cos(rz/2) - q3*sin(rz/2) + cos(rx/2)*cos(ry/2))/(sin(rz/2)*cos(rz/2))
#из (2) sin(rx/2)*cos(ry/2)=(q1 - sin(ry/2)*sin(rz/2)*cos(rx/2))/cos(rz/2)
#из (3) sin(rx/2)*cos(ry/2)=(-q2 + sin(ry/2)*cos(rx/2)*cos(rz/2))/sin(rz/2)
#приравниваем eq5=(q1 - sin(ry/2)*sin(rz/2)*cos(rx/2))/cos(rz/2)-(-q2 + sin(ry/2)*cos(rx/2)*cos(rz/2))/sin(rz/2)
((q1*sin(rz/2) - sin(ry/2)*sin(rz/2)**2*cos(rx/2))-(-q2*cos(rz/2) + sin(ry/2)*cos(rx/2)*cos(rz/2)**2))/(cos(rz/2)*sin(rz/2))
(q1*sin(rz/2) +q2*cos(rz/2) - sin(ry/2)*sin(rz/2)**2*cos(rx/2)-sin(ry/2)*cos(rx/2)*cos(rz/2)**2)/(cos(rz/2)*sin(rz/2))
(q1*sin(rz/2) +q2*cos(rz/2) - sin(ry/2)*cos(rx/2)*(sin(rz/2)**2)+cos(rz/2)**2)/(cos(rz/2)*sin(rz/2))
(q1*sin(rz/2) +q2*cos(rz/2) - sin(ry/2)*cos(rx/2))/(cos(rz/2)*sin(rz/2))
#упрощаем: (2*q1*sin(rz/2) + 2*q2*cos(rz/2) + sin(rx/2 - ry/2) - sin(rx/2 + ry/2))/sin(rz)
