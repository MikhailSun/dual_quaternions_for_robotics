import numpy as np
import makecharts as mc
import dual_quat_test as dqt
from dual_quat_test import DQ,link,frame
import matplotlib.pyplot as plt
from sympy import simplify

DH=dict(l1=dict(Tetta=0,d=0.507,a=0.265,alfa=90),
        l2=dict(Tetta=90,d=0,a=0.7,alfa=0),
        l3=dict(Tetta=0,d=0,a=0.02,alfa=90),
        l4=dict(Tetta=0,d=0.759,a=0,alfa=90),
        l5=dict(Tetta=0,d=0,a=0,alfa=-90),
        l6=dict(Tetta=180,d=0.143,a=0,alfa=0),)

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



# l1= link(DH['l1'])
# l1.transform(10.)
# l2=link(DH['l2'],origin_quat=l1.origin1)
# l2.transform(0.)
# l3=link(DH['l3'],origin_quat=l2.origin1)
# l3.transform(0.)
# l4=link(DH['l4'],origin_quat=l3.origin1)
# l4.transform(0.)
# l5=link(DH['l5'],origin_quat=l4.origin1)
# l5.transform(0.)
# l6=link(DH['l6'],origin_quat=l5.origin1)
# l6.transform(0.)
# o1=frame(base_point=[1,0,1])
# o1.rotate_by_axis(axis=[1,1,0],angle=10)
# o1.show()



# Q=dqt.DQ()
# Q.make_translation(xyz=[1.,0.,0.])
# P=dqt.DQ()
# P.make_rotation(axis_xyz=[0.,0.,1.],angle_dgr=0.)
#
# dq3=dqt.DQ()
# dq3.make_translation(xyz=[0.,0.,1.])
# dq4=dqt.DQ()
# dq4.make_rotation(axis_xyz=[0.,1.,0.],angle_dgr=0.)
#
# dq_res=DQ.dq_mult(Q,P)
# dq_res=DQ.dq_mult(dq_res,dq3)
# dq_res=DQ.dq_mult(dq_res,dq4)
# dq_res.print()

# x=0.5
# y=0.
# z=1.
# axis=[0.,0.,1]
# angle=45.
# pos=[x,y,z]
# X=[0.]
# Y=[0.]
# Z=[0.]
#
# q_rot=dqt.Q()
# q_rot.make_quat_from_angle_and_axis(angle=angle, axis=axis)
# du_quat=dqt.DQ()
# du_quat.set_by_Quat_and_Vect3(Qreal=q_rot , V3=pos)
# test_M=du_quat.dq_to_matrix()
# for line in test_M:
#     print(f'{line}')
# x=test_M[0][3]
# y=test_M[1][3]
# z=test_M[2][3]
# Q_rot=du_quat.dq_get_rot()
# Vect, Angle = Q_rot.q_to_axisangle()
# print(f'Q_rot={Q_rot}')
# print(f'Vect={Vect}   ANgle={Angle}')
# x_rot_axis=[0,Vect[0]]
# y_rot_axis=[0,Vect[1]]
# z_rot_axis=[0,Vect[2]]
# X.append(x)
# Y.append(y)
# Z.append(z)
#
# ax.plot(X, Y, Z, ls='-', marker='o', ms=0.5, lw=0.5)
# ax.text(x=x,y=y, z=z,s=f'x={round(x,3)}; y={round(y,3)}; z={round(z,3)}',fontsize='xx-small')
# ax.plot(x_rot_axis, y_rot_axis, z_rot_axis, ls='--', marker='o', ms=0.5, lw=0.5)
# Fig=mc.Chart(points_for_scatter=[],points_for_plot=[{'x':x,'y':y,'label':'121212'}],title='robot',xlabel='A',ylabel='B',color_ticklabels='yellow', color_fig='black', color_axes='gray',dpi=150,figure_size=(5,5))