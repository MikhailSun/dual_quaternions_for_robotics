import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as rt
import makecharts as mc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

axises=dict(x=(1.,0.,0.),
            y=(0.,1.,0.),
            z=(0.,0.,1.))

# Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def make_quat_from_angle_and_axis_lambda(angle,axis):
    basis=axises[axis]
    return (basis[0]*np.sin(np.radians(angle)/2),
            basis[1]*np.sin(np.radians(angle)/2),
            basis[2]*np.sin(np.radians(angle)/2),
            np.cos(np.radians(angle)/2),)

def make_quat_from_angle_and_axis(angle,axis):
    c1=0.0000000001 if np.sin(np.radians(angle)/2)==0 else np.sin(np.radians(angle)/2)
    return (axis[0]*c1,
            axis[1]*c1,
            axis[2]*c1,
            np.cos(np.radians(angle)/2),)

def get_angle_from_quat(q):
    return np.degrees(np.arccos(q[3])*2)

def make_quat_from_angle_and_axis_psi(angle,axis):
    return (axis[0]*np.radians(angle),
            axis[1]*np.radians(angle),
            axis[2]*np.radians(angle),
            0.,)


def q_mult(q1, q2): #первым выполняется действие последнего кватерниона! тут обратный порядок, как если бы мы последовательно сначала использовали последний кватернион, а потом первый!
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return x, y, z, w
def q_conjugate(q):
    x, y, z, w = q
    return (-x, -y, -z, w)

def qv_mult(q1, v1):
    q2 = np.append(v1,0.0)
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[:-1]

def norm_of_quat(q):
    norm= np.sqrt(sum(n * n for n in q))
    return norm

def inverse_quat(q):
    n=norm_of_quat(q)
    return q_conjugate(q)/n/n

def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = np.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v

def q_to_axisangle(q):
    v, w = q[:-1], q[-1]
    theta = np.arccos(w) * 2.0
    return normalize(v), np.degrees(theta)

def frame_rot_by_quat(X, Y, Z, q):
    X_rot=qv_mult(q, X)
    Y_rot = qv_mult(q, Y)
    Z_rot = qv_mult(q, Z)
    return X_rot, Y_rot, Z_rot

rot = rt.from_euler('z', 45, degrees=True)
quat = rot.as_quat()
my_quat=make_quat_from_angle_and_axis_lambda(45,'z')
print(f'True quat={quat}')
print(f"My quat={my_quat}")
p1=[1,0,0]
print(f'true prod={qv_mult(quat,p1)}')

#прямая кинематика:
#угол и ось(базис) -> make_quat_from_angle_and_axis_lambda = q
#вращение -> qv_mult(quat,точка) оно же общая формула: p1=q*p0*q^-1
def direct(axis, angle, point0_rel, base_point, base_quat, frame0):
    # print(f'Dir kin axis = {axis} angle={angle}')
    _q=make_quat_from_angle_and_axis(angle, axis)
    q_vect=make_quat_from_angle_and_axis_psi(angle, axis)
    q=q_mult(base_quat, _q)
    point1_rel=qv_mult(q, point0_rel)
    orient=qv_mult(q, axis)
    print(f'rel coord: {point0_rel} -> {point1_rel}')
    point1_abs=[x + y for x,y in zip(point1_rel, base_point)]
    point0_abs=[x + y for x, y in zip(point0_rel, base_point)]
    print(f'abs coord: {point0_abs} -> {point1_abs}')
    frame1=frame_rot_by_quat(frame0[0], frame0[1], frame0[2], q)
    return point1_abs, q, q_vect, orient, frame1

#здесь всегда будем подразумевать, что вращение происходит относительно оси Z в СК привязанной к линку по правилам Денавита_хартенберга
def direct_rel(QdaA,var_angle,base_frame):
    Q,d,a,const_angle=QdaA
    # 1) сначала крутим абс СК так, чтобы привести ее в соответствие с ДХ
    q1=make_quat_from_angle_and_axis(var_angle+Q,[0,0,1])
    q=make_quat_from_angle_and_axis(const_angle+var_angle,[0,0,1])



p0_rel=[0.265,0,0.485] #координаты движущейся СК относительно неподвижной
p1_rel=[0.,0.,0.7]
p2_rel=[0.6,0.,0.02]
p3_rel=[0.2,0.,0.]
p4_rel=[0.123,0.,0.]
p5_rel=[0.,0.,0.]
dp=[0,0,0] #смещение всех точекна указанные координаты
ax1=axises['z'] #ось относительно которой происходит вращение подвижн СК, ось задана в СК основания робота
ax2=axises['y']
ax3=axises['y']
ax4=axises['x']
ax5=axises['y']
ax6=axises['x']

base=[0,0,0]
root_quat=make_quat_from_angle_and_axis(1, [0, 0, 0])
frame_root=[[1, 0, 0],
            [0,1,0],
            [0,0,1]]
ang=[45.,45.,-50.,25.,45.,-45.] #ваьрируемый угол поворота
DH_QdaAlfa=[[0,0.507,0.265,-90],
            [-90,0,0.7,0],
            [0,0,20,-90],
            [0,0.759,0,90],
            [0,0,0,-90],
            [0,0,0,0]]
n_joints=len(DH_QdaAlfa)

# for i in n_joints:
#     direct_rel(DH_QdaAlfa[i],ang[i],)


p0_abs, p0_quat, p0_quat_vect, orient0, frame0=direct(ax1, ang[0], p0_rel, base, root_quat, frame_root)
p1_abs, p1_quat, p1_quat_vect, orient1, frame1=direct(ax2, ang[1], p1_rel, p0_abs, p0_quat, frame_root)
p2_abs, p2_quat, p2_quat_vect, orient2, frame2=direct(ax3, ang[2], p2_rel, p1_abs, p1_quat, frame_root)
p3_abs, p3_quat, p3_quat_vect, orient3, frame3=direct(ax4, ang[3], p3_rel, p2_abs, p2_quat, frame_root)
p4_abs, p4_quat, p4_quat_vect, orient4, frame4=direct(ax5, ang[4], p4_rel, p3_abs, p3_quat, frame_root)
p5_abs, p5_quat, p5_quat_vect, orient5, frame5=direct(ax6, ang[5], p5_rel, p4_abs, p4_quat, frame_root)
points=[base,p0_abs,p1_abs,p2_abs,p3_abs,p4_abs,p5_abs]
orients=[orient0,orient1,orient2,orient3,orient4,orient5]
frames=[frame_root, frame0, frame1, frame2, frame3, frame4, frame5]
X=[val[0] for val in points]
Y=[val[1] for val in points]
Z=[val[2] for val in points]
QUAT=[p0_quat,p1_quat,p2_quat,p3_quat,p4_quat,p5_quat]
AX_ANG=[q_to_axisangle(q) for q in QUAT]
# Fig=mc.Chart(points_for_plot=[{'x':X,'y':y,'label':'121212'}],title='111111',xlabel='A',ylabel='B',color_ticklabels='yellow', color_fig='black', color_axes='gray',dpi=150,figure_size=(5,5))
fig = plt.figure(dpi=150)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
# ax.clear()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1, 1, 1])  # IMPORTANT - this is the new, key line
# ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
# set_axes_equal(ax)
ax.plot(X, Y, Z, ls='-', marker='o', ms=0.5, lw=0.1)
#строим оси вращения в каждом сустве
for axis,point in zip(orients,points):
    L=0.05
    _x=[point[0]-axis[0]*L,point[0]+axis[0]*L]
    _y = [point[1]-axis[1]*L, point[1] + axis[1]*L]
    _z = [point[2]-axis[2]*L, point[2] + axis[2]*L]
    ax.plot(_x, _y, _z, ls='--', marker='', ms=1, lw=1, c='black')

#строим СК в суставах
for frame, point in zip(frames,points):
    scale=0.2
    _x=[point[0],scale*frame[0][0]+point[0]]
    _y = [point[1], scale*frame[0][1] + point[1]]
    _z = [point[2], scale*frame[0][2] + point[2]]
    ax.plot(_x,_y,_z, ls='-', marker='', ms=1, lw=0.5, c='red')
    _x=[point[0],scale*frame[1][0]+point[0]]
    _y = [point[1], scale*frame[1][1] + point[1]]
    _z = [point[2], scale*frame[1][2] + point[2]]
    ax.plot(_x,_y,_z, ls='-', marker='', ms=1, lw=0.5, c='green')
    _x=[point[0],scale*frame[2][0]+point[0]]
    _y = [point[1], scale*frame[2][1] + point[1]]
    _z = [point[2], scale*frame[2][2] + point[2]]
    ax.plot(_x,_y,_z, ls='-', marker='', ms=1, lw=0.5, c='blue')

#проверяем найдем ли мы точку p5 из точки p6 через conjugate quaternion
inv_q56=inverse_quat(make_quat_from_angle_and_axis(ang[5],ax6))
conj_q56=q_conjugate(make_quat_from_angle_and_axis(ang[5],ax6))
#если прямая кинематика p5=(q45*q14)*p4*conj(q45*q14)

# p4_test_inv=qv_mult(inv_q56,p5_abs)
# p4_test2_conj=qv_mult(conj_q56,p5_abs)
# X_test=[p4_test_inv[0],p5_abs[0]]
# Y_test=[p4_test_inv[1],p5_abs[1]]
# Z_test=[p4_test_inv[2],p5_abs[2]]
# ax.plot(X_test,Y_test, Z_test, ls='-', marker='', ms=1, lw=1, c='red')
# X_test=[p4_test2_conj[0],p5_abs[0]]
# Y_test=[p4_test2_conj[1],p5_abs[1]]
# Z_test=[p4_test2_conj[2],p5_abs[2]]
# ax.plot(X_test,Y_test, Z_test, ls='--', marker='', ms=1, lw=0.7, c='blue')
#
# q_test=make_quat_from_angle_and_axis(ang[5],ax6)
# q_inv_test=inverse_quat(q_test)


# q_to_axisangle(p2_quat)


def inverse_kin(axis, point0_rel, point1_rel):
    e1,e2,e3=axises[axis]
    x0,y0,z0=point0_rel
    x1, y1, z1 = point1_rel
    c1=e1**2*x0 + 2*e1*e2*y0 + 2*e1*e3*z0 - e2**2*x0 - e3**2*x0 + x0 - 2*x1
    c2 = 2*np.sqrt(e1**2*x0**2 - e1**2*x0*x1 + 2*e1*e2*x0*y0 - 2*e1*e2*x1*y0 + 2*e1*e3*x0*z0 - 2*e1*e3*x1*z0 - e2**2*x0**2 + e2**2*x0*x1 - e2**2*z0**2 + 2*e2*e3*y0*z0 - e3**2*x0**2 + e3**2*x0*x1 - e3**2*y0**2 - x0*x1 + x1**2,dtype=np.complex)
    c3=e1**2*x0 + 2*e1*e2*y0 + 2*e1*e3*z0 - e2**2*x0 + 2*1j*e2*z0 - e3**2*x0 - 2*1j*e3*y0 - x0
    a1=-1j*np.log((c1-c2)/c3)
    a2 = -1j * np.log((c1 + c2) / c3)
    print(f"a1={a1} a2={a2}")
    return a1,a2

def inverse_kin2(axis, point0_rel, point1_rel):
    e1, e2, e3 = axis
    x0,y0,z0=point0_rel
    X, Y, Z = point1_rel
    results=[]
    c1=-X**2 + X*e1**2*x0 + 2*X*e1*e2*y0 + 2*X*e1*e3*z0 - X*e2**2*x0 - X*e3**2*x0 + X*x0 - e1**2*x0**2 - 2*e1*e2*x0*y0 - 2*e1*e3*x0*z0 + e2**2*x0**2 + e2**2*z0**2 - 2*e2*e3*y0*z0 + e3**2*x0**2 + e3**2*y0**2
    c2=X - e1**2*x0 - 2*e1*e2*y0 - 2*e1*e3*z0 + e2**2*x0 + e3**2*x0
    c3=e2*z0 - e3*y0
    if (c2!=0):
        res1 = -2 * np.arctan((-c3 + np.sqrt(c1)) / (c2))
        res2 = 2 * np.arctan((c3 + np.sqrt(c1)) / (c2))
        results.append(res1)
        results.append(res2)
    c4=-Y**2 - Y*e1**2*y0 + 2*Y*e1*e2*x0 + Y*e2**2*y0 + 2*Y*e2*e3*z0 - Y*e3**2*y0 + Y*y0 + e1**2*y0**2 + e1**2*z0**2 - 2*e1*e2*x0*y0 - 2*e1*e3*x0*z0 - e2**2*y0**2 - 2*e2*e3*y0*z0 + e3**2*x0**2 + e3**2*y0**2
    c5=Y + e1**2*y0 - 2.0*e1*e2*x0 - e2**2*y0 - 2.0*e2*e3*z0 + e3**2*y0
    c6=e1*z0 - e3*x0
    if (c5!=0):
        res3 = 2.0 * np.arctan((-c6 + np.sqrt(c4)) / (c5))
        res4 = -2.0 * np.arctan((c6 + np.sqrt(c4)) / (c5))
        results.append(res3)
        results.append(res4)
    c7=-Z**2 - Z*e1**2*z0 + 2*Z*e1*e3*x0 - Z*e2**2*z0 + 2*Z*e2*e3*y0 + Z*e3**2*z0 + Z*z0 + e1**2*y0**2 + e1**2*z0**2 - 2*e1*e2*x0*y0 - 2*e1*e3*x0*z0 + e2**2*x0**2 + e2**2*z0**2 - 2*e2*e3*y0*z0 - e3**2*z0**2
    c8=Z + e1**2*z0 - 2*e1*e3*x0 + e2**2*z0 - 2*e2*e3*y0 - e3**2*z0
    c9=e1*y0 - e2*x0
    if (c8!=0):
        res5 = -2 * np.arctan((-c9 + np.sqrt(c7)) / c8)
        res6 = 2 * np.arctan((c9 + np.sqrt(c7)) / c8)
        results.append(res5)
        results.append(res6)
    return [np.degrees(x) for x in results]



#обратная кинематика:
#из общая формулы p1_rel=q*p0_rel*q^-1 нужно найти угол
#p1(x1,y1,z1,0)=q(e1*sin(a/2),e2*sin(a/2),e3*sin(a/2),cos(a/2))*p0(из ДХ-параметров)*q^-1
# print('check inverse kinematic')
# p0_rel=[0.5,0,0]
# base=[0,0,0]
# axis0_rel=axises['z']
# p1_abs, p1_quat, p0_quat_vect, orient0=direct(axis0_rel,1,p0_rel,base,make_quat_from_angle_and_axis(0,[0,0,0]))
# p1_rel=[0.5,0,0]
# axis1_rel=axises['y']
# p2_abs, p2_quat, p1_quat_vect, orient1=direct(axis1_rel,5,p1_rel,p1_abs,p1_quat)

# p1_rel=[x-y for x,y in zip(p1_abs, base)]
# res1=inverse_kin2(axis0_rel, p0_rel, [x-y for x,y in zip(p1_abs, base)])
# print(res1)
# res2=inverse_kin2(axis1_rel, p1_rel, [x-y for x,y in zip(p2_abs, p1_abs)])
# print(res2)
# a1,a2=inverse_kin(ax,p0,p1)
pass