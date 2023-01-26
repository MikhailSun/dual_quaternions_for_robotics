import numpy as np
# from dual_quat_test import DQ,Q
# from numpy import quaternion
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import makecharts as mc
import dual_quat_test as my_dq

import matplotlib.pyplot as plt

cmap = plt.get_cmap('jet')


# from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(dpi=150)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
# ax.clear()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

test_point=np.array([1,0.,0]) #тестовая точка
#ориентация 1
axis1=np.array([1, 1, 1]) #0.683012701892219 + 0.683012701892219*i + 0.183012701892219*j + 0.183012701892219*k
angle1=np.radians(1) #angle=30 x=0.229 y=0.132 z=0.507
Q1=my_dq.Q()
Q1.make_quat_from_angle_and_axis(angle=angle1,axis=axis1)
#ориентация 2
axis2=np.array([1, 1, 0]) #0.579227965339569 + 0.579227965339569*i + 0.405579787672639*j + 0.405579787672639*k
angle2=np.radians(1) #angle=70 x=0.091 y=0.249 z=0.507
Q2=my_dq.Q()
Q2.make_quat_from_angle_and_axis(angle=angle2,axis=axis2)

#задание начальной и конечной ориентации через вектор вращения
r1 = angle1 * axis1
r2 = angle2 * axis2
key_rots = R.from_rotvec([r1, r2])
key_times = [0, 1]
#расчет  слерп интерполяции
slerp = Slerp(key_times, key_rots)

N=10
times = np.linspace(0,1,N)
colors = [cmap(i) for i in np.linspace(0, 1, N)]

#
interp_rots = slerp(times)
interp_rots_as_quat=interp_rots.as_quat()
interp_rots_as_rotvec=interp_rots.as_rotvec()
interp_rots_as_matrix=interp_rots.as_matrix()
Y_angle_for_plot=[]
X_=[]

test1=my_dq.Slerp(Q1, Q2, 1)
test2=my_dq.Slerp(Q1, Q2, 0.0001)
my_slerps=[my_dq.Slerp(Q1, Q2, t) for t in times]

for i,m in enumerate(interp_rots_as_quat):
    m_abs=np.sqrt(m[0]**2+m[1]**2+m[2]**2)
    Y_angle_for_plot.append(np.degrees(np.arccos(m[3])*2))
    Xaxis=[0., m[0]/m_abs]
    Yaxis = [0., m[1]/m_abs]
    Zaxis = [0., m[2]/m_abs]
    ax.plot(Xaxis, Yaxis, Zaxis, ls=':', marker='o', ms=2, lw=1, c=colors[i])

# mc.Chart(points_for_plot=[{'x':x,'y':y,'label':'121212'}],title='111111',xlabel='A',ylabel='B',color_ticklabels='yellow', color_fig='black', color_axes='gray',dpi=150,figure_size=(5,5))

#
for i,q in enumerate(interp_rots):
    test_point_rotated=q.apply(test_point)
    print(test_point_rotated)
    ax.plot([0,test_point_rotated[0]], [0,test_point_rotated[1]], [0,test_point_rotated[2]], ls='-', marker='o', ms=2, lw=1, c=colors[i])


#
# orient1=Q(w=1, x=0, y=0, z=0)
# orient1.make_quat_from_angle_and_axis(angle=10, axis=[1.,1.,0.])



# orient2=Q(w=1, x=0, y=0, z=0)
# orient2.make_quat_from_angle_and_axis(angle=-10, axis=[1.,0.,0.])

# slerp=q1*(q1^-1*q2)^t #https://habr.com/ru/post/426863/
# q^t=exp(t*log(q)) = norm(q)*(cos(t*fi)+n*sin(t*fi))
# q^t = norm(q)^t * (cos(x*fi) + unit_vect*sin(x*fi) )
