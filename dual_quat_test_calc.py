import numpy as np
import makecharts as mc
import dual_quat_test as dqt
from dual_quat_test import DQ,link,frame
import matplotlib.pyplot as plt
from sympy import simplify, nsimplify, Derivative, diff,expand,sin,cos

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

