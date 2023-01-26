import numpy as np
# test of dual quaternion working
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import Quaternion as spQ
from sympy import symbols,sin,cos,sqrt,nan,simplify,nsimplify,Derivative,diff,expand,conjugate,Matrix,Function,acos
import copy

fig = plt.figure(dpi=150)
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
# ax.clear()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_box_aspect([1, 1, 1])
# w_, x_, y_,z_=symbols("w_, x_, y_,z_")
t=symbols('t')#переменная времени

class Q():
    number_of_Q=0
    def __init__(self, w=np.nan, x=np.nan, y=np.nan, z=np.nan, w_=nan, x_=nan, y_=nan, z_=nan):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self.Q = list([self.w, self.x, self.y, self.z])

        #для симпай
        Q.number_of_Q+=1
        self.N=str(Q.number_of_Q)
        # self.w_=symbols(f'w_q{self.N}')
        # self.x_ = symbols(f'x_q{self.N}')
        # self.y_ = symbols(f'y_q{self.N}')
        # self.z_ = symbols(f'z_q{self.N}')
        self.Q_=spQ(w_, x_, y_, z_)

    def __repr__(self):
        str1 = f'w={self.w}   x={self.x}   y={self.y}   z={self.z}'
        str2 = f'   norm={self.q_norm()}'
        return str1 + str2

    def make_quat_from_angle_and_axis(self, angle=0., axis=(0.,0.,0.), angle_=0., axis_=(0.,0.,0.)):
        # c1 = 0.0000000001 if np.sin(np.radians(angle) / 2) == 0 else np.sin(np.radians(angle) / 2)
        c1=np.sin(np.radians(angle) / 2)
        self.w = np.cos(np.radians(angle) / 2)
        axis, axis_ = Q.v_normalize(axis, axis_)
        self.x = axis[0] * c1
        self.y = axis[1] * c1
        self.z = axis[2] * c1
        self.Q = list([self.w, self.x, self.y, self.z])

        #для симпай
        # c1_ = sin(angle_ / 2)
        # self.Q_.a=cos(angle_/2)
        # self.Q_.b =axis_[0] * c1_
        # self.Q_.c =axis_[1] * c1_
        # self.Q_.d =axis_[2] * c1_
        self.Q_=self.Q_.from_axis_angle(vector=tuple(axis_) ,angle=angle_)

        return self

    def get_angle_from_quat(self):
        return np.degrees(np.arccos(self.w) * 2)

    def make_quat_from_angle_and_axis_psi(self, angle, axis):
        c1 = np.radians(angle)
        w = 0.
        x = axis[0] * c1
        y = axis[1] * c1
        z = axis[2] * c1
        self.__init__(w=w, x=x, y=y, z=z)
        # return self.w, self.x, self.y, self.z

    # первым выполняется действие последнего кватерниона! тут обратный порядок, как если бы мы последовательно сначала использовали последний кватернион, а потом первый!
    @staticmethod
    def q_mult(q1, q2):
        x1, y1, z1, w1 = q1.x, q1.y, q1.z, q1.w
        x2, y2, z2, w2 = q2.x, q2.y, q2.z, q2.w
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return Q(w=w, x=x, y=y, z=z)

    @staticmethod
    def q_sum(q1, q2):
        x1, y1, z1, w1 = q1.x, q1.y, q1.z, q1.w
        x2, y2, z2, w2 = q2.x, q2.y, q2.z, q2.w
        w = w1 + w2
        x = x1 + x2
        y = y1 + y2
        z = z1 + z2
        return Q(w=w, x=x, y=y, z=z)

    @staticmethod
    def q_conjugate(q):
        if isinstance(q,spQ):
            x, y, z, w = q.b, q.c, q.d, q.a
        else:
            x, y, z, w = q.x, q.y, q.z, q.w
        return Q(w=w, x=-x, y=-y, z=-z)

    @staticmethod
    def qv_mult(q1, v1):
        q2 = np.append(v1, 0.0)
        res = Q.q_mult(Q.q_mult(q1, q2), Q.q_conjugate(q1))
        return res.x, res.y, res.z

    @staticmethod
    def qs_mult(q, scalar):
        return Q(w=q.w * scalar, x=q.x * scalar, y=q.y * scalar, z=q.z * scalar)

    def q_norm(self):
        v = (self.w, self.x, self.y, self.z)
        # test = np.sqrt(self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)
        return np.sqrt(sum(float(x * x) for x in v))

    @staticmethod
    def q_inverse(q):
        n = Q.q_norm(q) ** 2
        q_conj = Q.q_conjugate(q)
        return Q(w=q_conj.w / n, x=q_conj.x / n, y=q_conj.y / n, z=q_conj.z / n)

    def normalize(self, tolerance=0.00001):
        norm = self.q_norm()
        if abs(norm - 1.0) > tolerance:
            self.w = self.w / norm
            self.x = self.x / norm
            self.y = self.y / norm
            self.z = self.z / norm
        return self

    @staticmethod
    def q_log(q):
        #log(q)=[0;Tetta*v]
        Tetta=np.arccos(q.w)
        unit_v=np.sqrt(q.x**2+q.y**2+q.z**2)
        return Q(w=0., x=Tetta*q.x/unit_v, y=Tetta*q.y/unit_v, z=Tetta*q.z/unit_v)

    @staticmethod
    def q_exp(q):
        #exp([0;Tetta*v]) = [cos(Tetta); sin(Tetta)*v]
        if (q.w!=0):
            print('Error!')
            raise SystemExit
        Tetta = np.sqrt(q.x ** 2 + q.y ** 2 + q.z ** 2)

        return Q(w=np.cos(Tetta), x=np.sin(Tetta)*q.x/Tetta, y=np.sin(Tetta)*q.y/Tetta, z=np.sin(Tetta)*q.z/Tetta)

    @staticmethod
    def q_pow(q,t):
        #q^t = exp(t*log(q))
        log_q=Q.q_log(q).normalize()
        t_mult_log_q=Q.qs_mult(log_q,t)
        return Q.q_exp(t_mult_log_q)


    @staticmethod
    def v_normalize(v,v_sp=None):
        norm = np.sqrt(sum(x * x for x in v))
        res = []
        for val in v:
            res.append(val / norm)

        #for sympy
        if (not v_sp is None):
            norm_ = sqrt(sum(x * x for x in v_sp))
            res_ = []
            for val in v_sp:
                res_.append(val / norm_)
            return res, res_
        else:
            return res

    def q_to_axisangle(self):
        v, w = [self.x, self.y, self.z], self.w
        theta = np.arccos(w) * 2.0
        return Q.v_normalize(v), np.degrees(theta)

    def q_to_coordinates(self):
        return self.x, self.y, self.z

    def frame_rot_by_quat(self, frame_matrix):
        X_rel = [frame_matrix[0][0], frame_matrix[1][0], frame_matrix[2][0]]
        Y_rel = [frame_matrix[0][1], frame_matrix[1][1], frame_matrix[2][1]]
        Z_rel = [frame_matrix[0][2], frame_matrix[1][2], frame_matrix[2][2]]
        X_rot = Q.qv_mult(self, X_rel)
        Y_rot = Q.qv_mult(self, Y_rel)
        Z_rot = Q.qv_mult(self, Z_rel)
        return [[X_rot[0], Y_rot[0], Z_rot[0]],
                [X_rot[1], Y_rot[1], Z_rot[1]],
                [X_rot[2], Y_rot[2], Z_rot[2]]]

    # https://stackoverflow.com/questions/5782658/extracting-yaw-from-a-quaternion#:~:text=Having%20given%20a%20Quaternion%20q,*q.y%20%2D%20q.z*q.z)%3B
    # var yaw = atan2(2.0*(q.y*q.z + q.w*q.x), q.w*q.w - q.x*q.x - q.y*q.y + q.z*q.z);
    # var pitch = asin(-2.0*(q.x*q.z - q.w*q.y));
    # var roll = atan2(2.0*(q.x*q.y + q.w*q.z), q.w*q.w + q.x*q.x - q.y*q.y - q.z*q.z);
    def get_rxryrz(self):
        # This should fit for intrinsic tait - bryan rotation of xyz-order.For other rotation orders, extrinsic and proper-euler rotations other conversions have to be used.
        yaw = np.arctan2(2.0*(self.y*self.z + self.w*self.x), self.w*self.w - self.x*self.x - self.y*self.y + self.z*self.z)
        pitch = np.arcsin(-2.0*(self.x*self.z - self.w*self.y))
        roll = np.arctan2(2.0*(self.x*self.y + self.w*self.z), self.w*self.w + self.x*self.x - self.y*self.y - self.z*self.z)
        print(f'rx={np.degrees(yaw)} ry={np.degrees(pitch)} rz={np.degrees(roll)}')

        # I've verified below code against Wikipedia's equations plus Pixhawk's documentation and it is correct.
        # If you are working with drones/aviation, below is the code (taken directly from DJI SDK). Here q0, q1, q2, q3 corresponds to w,x,y,z components of the quaternion respectively. Also note that yaw, pitch, roll may be referred to as heading, attitude and bank respectively in some literature.
        roll = np.arctan2(2.0 * (self.z * self.y + self.w * self.x), 1.0 - 2.0 * (self.x * self.x + self.y * self.y))
        pitch = np.arcsin(2.0 * (self.y * self.w - self.z * self.x))
        yaw = np.arctan2(2.0 * (self.z * self.w + self.x * self.y), - 1.0 + 2.0 * (self.w * self.w + self.x * self.x))
        print(f'rx={np.degrees(yaw)} ry={np.degrees(pitch)} rz={np.degrees(roll)}')

        #Yaw, Pitch, Roll is the same as Heading, Pitch, Bank.
        yaw = np.arctan2(2 * self.y * self.w - 2 * self.x * self.z, 1 - 2 * self.y*self.y - 2 * self.z*self.z)
        pitch = np.arcsin(2 * self.x * self.y + 2 * self.z * self.w)
        roll = np.arctan2(2 * self.x * self.w - 2 * self.y * self.z, 1 - 2 * self.x*self.x - 2 * self.z*self.z)

        # print(f'yaw={np.degrees(yaw)} pitch={np.degrees(pitch)} roll={np.degrees(roll)}')
        print(f'rx={np.degrees(yaw)} ry={np.degrees(pitch)} rz={np.degrees(roll)}')
        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

    def diff_(self):
        return self.Q_.diff()

    def diff(self):
        #тут должна быть штука для численного вычисления производной кватерниона
        pass

class DN():
    number_of_DN=0
    def __init__(self, real=0., dual=0., real_=0., dual_=0.):
        self.m_real = real
        self.m_dual = dual

        #для симпай
        DN.number_of_DN+=1
        N=str(DN.number_of_DN)
        self.N=N
        # self.wr_=symbols(f'wr{N}')
        # self.xr_ = symbols(f'xr{N}')
        # self.yr_ = symbols(f'yr{N}')
        # self.zr_ = symbols(f'zr{N}')
        # self.wd_=symbols(f'wd{N}')
        # self.xd_ = symbols(f'xd{N}')
        # self.yd_ = symbols(f'yd{N}')
        # self.zd_ = symbols(f'zd{N}')
        self.m_real_=real_
        self.m_dual_=dual_

class DQ():
    number_of_DQ=0
    def __init__(self, Qreal=Q(w=1, x=0, y=0, z=0), Qdual=Q(w=0, x=0, y=0, z=0), Qreal_=spQ(a=1, b=0, c=0, d=0), Qdual_=spQ(a=0, b=0, c=0, d=0)):
        self.m_real = Qreal.normalize()
        self.m_dual = Qdual

        #для симпай
        DQ.number_of_DQ+=1
        N=str(DQ.number_of_DQ)
        self.N=N
        # self.wr_=symbols(f'wr{N}')
        # self.xr_ = symbols(f'xr{N}')
        # self.yr_ = symbols(f'yr{N}')
        # self.zr_ = symbols(f'zr{N}')
        # self.wd_=symbols(f'wd{N}')
        # self.xd_ = symbols(f'xd{N}')
        # self.yd_ = symbols(f'yd{N}')
        # self.zd_ = symbols(f'zd{N}')
        self.m_real_=Qreal_.normalize()
        self.m_dual_=Qdual_

    def __repr__(self):
        to_print='Cyfral:\n'
        to_print+= f'real:\n'
        to_print+=self.m_real.__repr__() + '\n'
        to_print+= f'dual:\n'
        to_print+=self.m_dual.__repr__() + '\n'
        to_print+='Analytical\n'
        to_print+= f'real:\n'
        to_print+=str(self.m_real_)+ '\n'
        to_print+= f'dual:\n'
        to_print+= str(self.m_dual_) + '\n'
        return to_print

    def simplify(self):
        # self.m_real_=nsimplify(simplify(self.m_real_), tolerance=1e-10)
        # self.m_dual_ = nsimplify(simplify(self.m_dual_), tolerance=1e-10)
        self.m_real_=nsimplify(self.m_real_,  rational=False)#tolerance=1e-10,
        self.m_dual_ = nsimplify(self.m_dual_,  rational=False)#tolerance=1e-10,

    def insert_numbers(self,list_of_tuples_of_numbers):
        self.m_real_=self.m_real_.subs(list_of_tuples_of_numbers)
        self.m_dual_=self.m_dual_.subs(list_of_tuples_of_numbers)

    def copy_from_dq(self,dq):
        self.m_real=dq.m_real
        self.m_dual = dq.m_dual
        self.m_real_=dq.m_real_
        self.m_dual_ = dq.m_dual_

    def set_by_Quat_and_Vect3(self, Qreal=Q(w=1, x=0, y=0, z=0), V3=(0., 0., 0.), Qreal_=spQ(a=1, b=0, c=0, d=0), V3_=(0., 0., 0.)):
        self.m_real = Qreal.normalize()
        # self.m_dual =(Q.make_quat_from_angle_and_axis(0.,V3)*self.m_real)*0.5\
        # test=Q.make_quat_from_angle_and_axis(0., V3)
        q = Q(w=0., x=V3[0], y=V3[1], z=V3[2])
        self.m_dual = Q.qs_mult(Q.q_mult(q, self.m_real), 0.5)

        self.m_real_ = Qreal_.normalize()
        q_ = spQ(a=0., b=V3_[0], c=V3_[1], d=V3_[2])
        self.m_dual_ =q_.mul(self.m_real_).mul(0.5)

    def make_dual_quat_from_coord_axis_angle(self, xyz=(0., 0., 0.), axis_xyz=(1., 0., 0.), angle_dgr=0., xyz_=(0., 0., 0.), axis_xyz_=(1., 0., 0.), angle_dgr_=0):
        q = Q()
        q.make_quat_from_angle_and_axis(angle=angle_dgr, axis=axis_xyz, angle_=angle_dgr_, axis_=axis_xyz_)

        # q_=spQ()
        # q_.from_axis_angle(vector=axis_xyz_, angle=angle_dgr_)

        self.set_by_Quat_and_Vect3(Qreal=q, V3=xyz, Qreal_=q.Q_, V3_=xyz_)

    def make_translation(self, xyz, xyz_):
        self.make_dual_quat_from_coord_axis_angle(xyz=xyz, xyz_=xyz_)

    def make_rotation(self, axis_xyz, angle_dgr, axis_xyz_, angle_dgr_):
        self.make_dual_quat_from_coord_axis_angle(axis_xyz=axis_xyz, angle_dgr=angle_dgr, axis_xyz_=axis_xyz_, angle_dgr_=angle_dgr_)

    # public static float Dot(DualQuaternion_c a, DualQuaternion_c b )
    # {
    # return Quaternion.Dot( a.m_real, b.m_real );
    # }
    # // Multiplication order - left to right
    # public static DualQuaternion_c operator * (DualQuaternion_c lhs, DualQuaternion_c rhs)
    # {
    # return new DualQuaternion_c(rhs.m_real * lhs.m_real, rhs.m_dual * lhs.m_real + rhs.m_real * lhs.m_dual);
    # }
    # порядок действия множителей: сначала выполняется трансформация из P, потом из Q
    @staticmethod
    def dq_mult(dq1, dq2):
        # формула: qr1*qr2+(qr1*qd2+qd1*qr2)*eps
        Qr = dq2.m_real
        Pr = dq1.m_real
        Qd = dq2.m_dual
        Pd = dq1.m_dual
        # print(f'TEST dq_mult: a.m_real={Q.m_real}, b.m_real={P.m_real}')
        return DQ(Qreal=Q.q_mult(Pr,Qr), Qdual=Q.q_sum(Q.q_mult(Pr, Qd), Q.q_mult(Pd, Qr)))

    @staticmethod
    def dq_mult_(dq1, dq2):
        # формула: qr1*qr2+(qr1*qd2+qd1*qr2)*eps
        Qr = dq2.m_real_
        Pr = dq1.m_real_
        Qd = dq2.m_dual_
        Pd = dq1.m_dual_
        # print(f'TEST dq_mult: a.m_real={Q.m_real}, b.m_real={P.m_real}')
        return DQ(Qreal_=Pr.mul(Qr), Qdual_=Pr.mul(Qd).add(Pd.mul(Qr)))

    def post_mult_by_dq(self,dq):
        Pr = self.m_real
        Pd = self.m_dual
        Qr = dq.m_real
        Qd = dq.m_dual
        self.m_real=Q.q_mult(Pr,Qr)
        self.m_dual=Q.q_sum(Q.q_mult(Pr, Qd), Q.q_mult(Pd, Qr))

        Pr_ = self.m_real_
        Pd_ = self.m_dual_
        Qr_ = dq.m_real_
        Qd_ = dq.m_dual_
        self.m_real_=Pr_.mul(Qr_)
        self.m_dual_=Pr_.mul(Qd_).add(Pd_.mul(Qr_))
        return self

    @staticmethod
    def dq_scalar_mult(dq, scalar):
        return DQ(Qreal=Q.qs_mult(dq.m_real, scalar), Qdual=Q.qs_mult(dq.m_dual, scalar), Qreal_=dq.m_real_*2, Qdual_=dq.m_dual_*2)

    @staticmethod
    def dn_mult_dq(dn,dq):
        #умножение дуального числа на бикватернион (коммутативно) - Гордеев, стр139, п4,2, п.4
        dn_real=dn.m_real
        dn_dual = dn.m_dual
        dq_real=dq.m_real
        dq_dual=dq.m_dual

        dn_real_=dn.m_real_
        dn_dual_ = dn.m_dual_
        dq_real_=dq.m_real_
        dq_dual_=dq.m_dual_
        return DQ(Qreal=Q(w=dn_real*dq_real.w,
                   x=dn_real*dq_real.x,
                   y=dn_real*dq_real.y,
                   z=dn_real*dq_real.z),
           Qdual=Q(w=dn_real*dq_dual.w+dn_dual*dq_real.w,
                   x=dn_real*dq_dual.x+dn_dual*dq_real.x,
                   y=dn_real*dq_dual.y+dn_dual*dq_real.y,
                   z=dn_real*dq_dual.z+dn_dual*dq_real.z),
           Qreal_=spQ(a=dn_real_ * dq_real_.a,
                   b=dn_real_ * dq_real_.b,
                   c=dn_real_ * dq_real_.c,
                   d=dn_real_ * dq_real_.d),
           Qdual_=spQ(a=dn_real_ * dq_dual_.a + dn_dual_ * dq_real_.a,
                      b=dn_real_ * dq_dual_.b + dn_dual_ * dq_real_.b,
                      c=dn_real_ * dq_dual_.c + dn_dual_ * dq_real_.c,
                      d=dn_real_ * dq_dual_.d + dn_dual_ * dq_real_.d))


    @staticmethod
    def dq_conjugate(dq):
        return DQ(Qreal=Q.q_conjugate(dq.m_real), Qdual=Q.q_conjugate(dq.m_dual))

    @staticmethod
    def dq_conjugate_(dq_):
        return DQ(Qreal_=spQ.conjugate(dq_.m_real_), Qdual_=spQ.conjugate(dq_.m_dual_))

    def dq_norm(self):
        # полная формула нормы дуального кватерниона какая-то непонятная, везде пишут, что вроде бы формула ниже - это для единичного дуального кватерниона
        # ?! откуда эта формула? норма(m_real)+eps*((m_real*m_dual)/норма(m_real)) = m_real*m_real_conj + eps((m_real*m_dual)/(m_real*m_real_conj))
        # или подругому: norm(DQ)=DQ*conj(DQ)=(DQreal+eps*DQdual)*(DQreal_conj+eps*DQdual_conj)=DQreal*DQreal_conj+eps*DQdual*DQreal_conj+DQreal*eps*DQdual_conj+eps*DQdual*eps*DQdual_conj =
        # DQreal*DQreal_conj + eps*(DQdual*DQreal_conj+DQreal*DQdual_conj). Total: norm(DQ) = (DQreal*DQreal_conj; DQdual*DQreal_conj+DQreal*DQdual_conj)
        dq1 = self
        dq2 = DQ.dq_conjugate(self)
        return DQ.dq_mult(dq1, dq2) #TODO! возможно здесь ошибка, должно возвращать дуальное число, проверить! Гордеев стр.138

    def dq_norm_(self):
        # полная формула нормы дуального кватерниона какая-то непонятная, везде пишут, что вроде бы формула ниже - это для единичного дуального кватерниона
        # ?! откуда эта формула? норма(m_real)+eps*((m_real*m_dual)/норма(m_real)) = m_real*m_real_conj + eps((m_real*m_dual)/(m_real*m_real_conj))
        # или подругому: norm(DQ)=DQ*conj(DQ)=(DQreal+eps*DQdual)*(DQreal_conj+eps*DQdual_conj)=DQreal*DQreal_conj+eps*DQdual*DQreal_conj+DQreal*eps*DQdual_conj+eps*DQdual*eps*DQdual_conj =
        # DQreal*DQreal_conj + eps*(DQdual*DQreal_conj+DQreal*DQdual_conj). Total: norm(DQ) = (DQreal*DQreal_conj; DQdual*DQreal_conj+DQreal*DQdual_conj)
        dq1_ = self
        dq2_ = DQ.dq_conjugate_(self)
        return DQ.dq_mult_(dq1_, dq2_)

    def dq_inverse(self):
        #Челноков, формула 1,105, стр70
        #обратный ДК равен сопряженный ДК деленный на норму
        dq_conj=DQ.dq_conjugate(self)
        dq_conj_=DQ.dq_conjugate_(self)
        dq_norma=self.dq_norm() #норма ДК - это дуальное число
        dq_norma_ = self.dq_norm_()
        test=DQ.dn_mult_dq(dq_norma,dq_conj)
        test_ = DQ.dn_mult_dq(dq_norma_, dq_conj_)
        # return DQ(Qreal=Q()) остановился тут нужно поделить конджугейт на норму и вернуть обратный ДК

    # public static DualQuaternion_c Normalize(DualQuaternion_c q)
    # {
    #     float mag = Quaternion.Dot(q.m_real, q.m_real);
    #     Debug_c.Assert(mag > 0.000001 f );
    #     DualQuaternion_c ret = q;
    #     ret.m_real *= 1.0f / mag;
    #     ret.m_dual *= 1.0f / mag;
    #     return ret;
    # }
    def normalize(self):
        # q*q_conj
        n = self.dq_norm().m_real.q_norm()
        # print(f'NEED to check: n must be scalar n.real={n.m_real} n.dual={n.m_dual}')
        self.m_real = Q.qs_mult(self.m_real, 1 / n)
        self.m_dual = Q.qs_mult(self.m_dual, 1 / n)

    def normalize_(self):
        # q*q_conj
        n = self.dq_norm_().m_real_.norm()
        # print(f'NEED to check: n must be scalar n.real={n.m_real} n.dual={n.m_dual}')
        self.m_real_ = self.m_real_.mul(1 / n)
        self.m_dual_ = self.m_dual_.mul(1 / n)

    @staticmethod
    def dq_sum(dq1, dq2):
        return DQ(Qreal=Q.q_sum(dq1.m_real, dq2.m_real), Qdual=Q.q_sum(dq1.m_dual, dq2.m_dual))

    def dq_get_rot(self):
        return self.m_real

    def dq_get_trans(self):
        q = Q.q_mult(Q.qs_mult(self.m_dual, 2.), Q.q_conjugate(self.m_real))
        return [q.x, q.y, q.z]

    def dq_to_matrix(self):
        self.normalize()
        M = [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.]]
        w = self.m_real.w
        x = self.m_real.x
        y = self.m_real.y
        z = self.m_real.z
        #из книги Modern Robotics, Lynch and Park, Cambridge U. Press, 2017., стр581
        #орт OX в глобальной системе координат
        M[0][0]=w*w+x*x-y*y-z*z
        M[1][0]=2*(w*z+x*y)
        M[2][0]=2*(x*z-w*y)
        # орт OY
        M[0][1]=2*(x*y-w*z)
        M[1][1]=w*w-x*x+y*y-z*z
        M[2][1]=2*(w*x+y*z)
        # орт OZ
        M[0][2]=2*(w*y+x*z)
        M[1][2]=2*(y*z-w*x)
        M[2][2]=w*w-x*x-y*y+z*z

        t = Q.q_mult(Q.qs_mult(self.m_dual, 2.), Q.q_conjugate(self.m_real))
        M[0][3] = t.x
        M[1][3] = t.y
        M[2][3] = t.z
        return M

    def dq_to_matrix_(self):
        self.normalize_()
        M = Matrix([[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.]])
        w = self.m_real_.a
        x = self.m_real_.b
        y = self.m_real_.c
        z = self.m_real_.d
        #из книги Modern Robotics, Lynch and Park, Cambridge U. Press, 2017., стр581
        M[0,0]=w*w+x*x-y*y-z*z
        M[1,0]=2*(w*z+x*y)
        M[2,0]=2*(x*z-w*y)
        M[0,1]=2*(x*y-w*z)
        M[1,1]=w*w-x*x+y*y-z*z
        M[2,1]=2*(w*x+y*z)
        M[0,2]=2*(w*y+x*z)
        M[1,2]=2*(y*z-w*x)
        M[2,2]=w*w-x*x-y*y+z*z
        # test=self.m_real_.to_rotation_matrix()

        t=self.m_dual_.mul(2.).mul(spQ.conjugate(self.m_real_))
        # t = Q.q_mult(Q.qs_mult(self.m_dual, 2.), Q.q_conjugate(self.m_real))
        M[0,3] = t.b
        M[1,3] = t.c
        M[2,3] = t.d
        return M

    def dq_to_coordinates(self):
        M=self.dq_to_matrix()
        x = M[0][3]
        y = M[1][3]
        z = M[2][3]
        return x,y,z

    def dq_to_axes(self):
        M = self.dq_to_matrix()
        x_axis = [M[0][0], M[1][0], M[2][0]]
        y_axis = [M[0][1], M[1][1], M[2][1]]
        z_axis = [M[0][2], M[1][2], M[2][2]]
        return x_axis, y_axis, z_axis

    def dq_to_frame(self):
        M = self.dq_to_matrix()
        x = M[0][3]
        y = M[1][3]
        z = M[2][3]
        x_axis = [M[0][0], M[1][0], M[2][0]]
        y_axis = [M[0][1], M[1][1], M[2][1]]
        z_axis = [M[0][2], M[1][2], M[2][2]]
        return frame(x_axis,y_axis,z_axis,(x,y,z))

    # def diff(self):
    #     dq1=

    def diff_(self):
        return DQ(Qreal_=self.m_real_.diff(),Qdual_=self.m_dual_.diff())

    def check(self):
        M = self.dq_to_matrix()
        x = M[0][3]
        y = M[1][3]
        z = M[2][3]
        x_axis = [M[0][0], M[1][0], M[2][0]]
        y_axis = [M[0][1], M[1][1], M[2][1]]
        z_axis = [M[0][2], M[1][2], M[2][2]]
        print(f'x={round(x,3)}, y={round(x,3)}, z={round(x,3)}')

    def print(self):
        M = self.dq_to_matrix()
        # for line in M:
        #     print(f'{line}')
        x = M[0][3]
        y = M[1][3]
        z = M[2][3]
        Vect, Angle = self.m_real.q_to_axisangle()
        # print(f'Q_rot={Q_rot}')
        # print(f'Vect={Vect}   ANgle={Angle}')
        scale = 0.2
        x_rot_axis = [-Vect[0] * scale, Vect[0] * scale]
        y_rot_axis = [-Vect[1] * scale, Vect[1] * scale]
        z_rot_axis = [-Vect[2] * scale, Vect[2] * scale]
        X = [0.]
        Y = [0.]
        Z = [0.]
        X.append(x)
        Y.append(y)
        Z.append(z)

        ax.plot(X, Y, Z, ls='-', marker='o', ms=0.5, lw=0.5)
        coords = f'x={round(x, 3)}; y={round(y, 3)}; z={round(z, 3)}'
        ang = f' Angle={round(Angle, 3)}'
        ax.text(x=x, y=y, z=z, s=coords + ang, fontsize='xx-small')
        ax.plot(x_rot_axis, y_rot_axis, z_rot_axis, ls='--', marker='.', ms=0.5, lw=0.5)
    def get_plukker_coordinates(self):
        #Гордеев стр154
        L=sqrt(self.m_real.x**2+self.m_real.y**2+self.m_real.z**2)
        real_part=[self.m_real.x/L,
                   self.m_real.y/L,
                   self.m_real.z/L]
        dual_part=[self.m_dual.x/L+self.m_real.w*self.m_real.x*self.m_dual.w/L/L/L,
                   self.m_dual.y/L+self.m_real.w*self.m_real.y*self.m_dual.w/L/L/L,
                   self.m_dual.z/L+self.m_real.w*self.m_real.z*self.m_dual.w/L/L/L]
        fi=np.degrees(2*np.arccos(self.m_real.w))
        fi0=-2*self.m_dual.w/L
        print(f'real part={real_part}\n'
              f'dual part={dual_part}\n'
              f'fi={fi}\n'
              f'fi0={fi0}')

        return real_part, dual_part

def Slerp(Q1, Q2, t):
    # slerp=q1*(q1^-1*q2)^t #https://habr.com/ru/post/426863/
    # q^t=exp(t*log(q)) = norm(q)*(cos(t*fi)+n*sin(t*fi))
    # q^t = norm(q)^t * (cos(x*fi) + unit_vect*sin(x*fi) )
    # see here: dam_eb_koch_m_lillholm_m_quaternions_interpolation_and_anima.pdf
    Q1inv=Q.q_inverse(Q1)
    Qtemp1=Q.q_mult(Q1inv,Q2)
    Qtemp2=Q.q_pow(Qtemp1,t)
    return Q.q_mult(Q1, Qtemp2)





class frame():
    number_of_link = 0
    def __init__(self, x=(1., 0., 0.), y=(0., 1., 0.), z=(0., 0., 1.), base_point=(0.,0.,0.)):
        self.axes={'x':x,
                     'y':y,
                     'z':z}
        self.base=base_point

    def set_from_dq(self,dq):
        x_ax, y_ax, z_ax=dq.dq_to_axes()
        base_pnt=dq.dq_to_coordinates()
        self.axes={'x':x_ax,
                     'y':y_ax,
                     'z':z_ax}
        self.base=base_pnt

    def rotate_by_axis(self,axis,angle):
        dq_rot = DQ()
        dq_rot.make_rotation(axis_xyz=axis, angle_dgr=float(angle))
        for key,_axis in self.axes.items():
            dq=DQ()
            dq.make_translation(_axis)
            dq=DQ.dq_mult(dq,dq_rot)
            self.axes[key]=dq.dq_to_coordinates()

    def rotate_by_dq(self,dq_rot):
        for key,axis in self.axes.items():
            dq=DQ()
            dq.make_translation(axis)
            dq=DQ.dq_mult(dq,dq_rot)
            self.axes[key]=dq.dq_to_coordinates()

    def translate_by_dq(self,dq_trans):
        self.base=dq_trans.dq_to_coordinates()

    def show(self,scale=0.3,ls='--',lw=0.5):
        colors=['red','blue','green']
        for key_axis,color in zip(self.axes.items(), colors):
            key,axis=key_axis
            x=self.base[0]+axis[0]*scale
            y=self.base[1] + axis[1]*scale
            z=self.base[2] + axis[2]*scale
            X=[self.base[0], x]
            Y = [self.base[1], y]
            Z = [self.base[2], z]
            ax.plot(X, Y, Z, ls=ls,  ms=0.5, lw=lw, c=color)
            # axes = f'x={round(x, 3)}; y={round(y, 3)}; z={round(z, 3)}'
            # ang = f' a={round(Angle, 3)}'
            # ax.text(x=x, y=y, z=z, s=axes, fontsize='xx-small')
            # ax.plot(x_rot_axis, y_rot_axis, z_rot_axis, ls='--', marker='.', ms=0.5, lw=0.5)
        coord_text=f'x={round(self.base[0], 3)}; y={round(self.base[1], 3)}; z={round(self.base[2], 3)}'
        ax.text(x=self.base[0], y=self.base[1], z=self.base[2], s=coord_text, fontsize='xx-small')



class link():
    number_of_link=0
    def __init__(self, DH_dict, origin_quat=None):
        link.number_of_link+=1
        self.N=str(link.number_of_link)
        # параметры ДХ
        self.Tetta_const = DH_dict['Tetta']
        self.d = DH_dict['d']
        self.a = DH_dict['a']
        self.alfa = DH_dict['alfa']  # NB! этот угол применяется только для поворота СК на конце текущего линка, но он не применяется для вычисления координат текущего линка - это вытекает из условий ДХ

        self.Tetta_const_ = symbols(f'Q_const{self.N}')
        self.d_ = symbols(f'd{self.N}')
        self.a_ = symbols(f'a{self.N}')
        self.alfa_ = symbols(f'A{self.N}')

        #СК начала и конца линка
        if origin_quat is None:
            self.origin0=DQ()
        else:
            self.origin0 = origin_quat
        self.origin1=DQ()

    #всегда в соответсвтие с ДХ подразумеваем, что вращение происходит относительно оси Z
    def transform(self, Tetta, V_Tetta=0., A_Tetta=0.):
        #NB! правила порядка умножения бикватернионов хорошо расписаны в "Гордеев. Кватернионы и бикватернионы с приложениями в геометрии и механике", стр.156, п.4,7,4
        # Tetta=np.radians(float(Tetta))

        # считаем положение линка в соответсвтии с правилами ДХ:
        # 1)вычисляем СК для начальной точки линка
        dq = DQ()
        # sp_Tetta=symbols(f'Q{self.N}', cls=Function)
        sp_Tetta=Function(f'Q{self.N}')(t)
        dq.make_rotation(axis_xyz=[0.,0.,1.], angle_dgr=float(Tetta) + self.Tetta_const, axis_xyz_=[0.,0.,1.], angle_dgr_=self.Tetta_const_+sp_Tetta) #из ДХ - вращаем относительно исходной оси z
        # self.origin0=DQ.dq_mult(self.origin0,dq)
        self.origin0.post_mult_by_dq(dq)
        #вспомогательные штуки для визуализации
        frame0=self.origin0.dq_to_frame()
        frame0.show(ls='-', lw=0.2)

        # 2)вычисляем координаты конца текущего линка, двигаем по ДХ после вращения на Tetta (п.1) на расстояния d и a
        dq_coord = DQ()
        dq_coord.make_translation(xyz=[self.a, 0., self.d], xyz_=[self.a_, 0., self.d_])
        self.origin1.copy_from_dq(self.origin0)
        self.origin1.post_mult_by_dq(dq_coord)
        # вспомогательные штуки для визуализации
        # frame1=dq.dq_to_frame()
        # frame1.show(lw=0.1)

        # 3)вычисляем координаты и ориентацию СК на конце текущего линка - она же будет базовой для последующего линка
        # по ДХ - вращаем относительно оси x текущей системы координат на угол alfa
        dq_rot = DQ()
        sp_Alfa = symbols(f'A{self.N}')
        dq_rot.make_rotation(axis_xyz=[1.,0.,0.], angle_dgr=self.alfa, axis_xyz_=[1.,0.,0.], angle_dgr_=self.alfa_) #из ДХ - вращаем относительно новой оси x
        self.origin1.post_mult_by_dq(dq_rot)
        frame1=self.origin1.dq_to_frame()
        frame1.show(ls=':',lw=0.5)
        line_X=[frame0.base[0],frame1.base[0]]
        line_Y = [frame0.base[1],frame1.base[1]]
        line_Z = [frame0.base[2],frame1.base[2]]
        plt.plot(line_X,line_Y,line_Z,lw=2,c='black')

        #подставляем в формулу параметры ДХ, но пока не подставляем переменные угол, угловую скорость и ускорение - т.о. получаем на выходе аналитиеские формулы для кватернионов
        self.origin0.insert_numbers([(f'A{self.N}',np.radians(self.alfa)),
                                        (f'Q_const{self.N}',np.radians(self.Tetta_const)),
                                        (f'd{self.N}',self.d),
                                        (f'a{self.N}',self.a),])
                                        # (f'Q{self.N}',np.radians(Tetta))])
        self.origin1.insert_numbers([(f'A{self.N}',np.radians(self.alfa)),
                                        (f'Q_const{self.N}',np.radians(self.Tetta_const)),
                                        (f'd{self.N}',self.d),
                                        (f'a{self.N}',self.a),])
                                        # (f'Q{self.N}',np.radians(Tetta))])
        #упрощаем формулу до читабельного вида и сохраняем ее копию в объекты frame0_dq (СК начала звена) и в frame1_dq (конец звена)
        # self.origin0.simplify()
        self.frame0_dq=copy.deepcopy(self.origin0)
        # self.origin1.simplify()
        # self.origin1_velocity_temp1=self.origin1.diff_() #TODO! нужно скалярно умножить на 2 и поделить на дуальный кв origin1 (т.е. умножит на ДК обратный origin1)
        #скалярно умножаем на 2
        # self.origin1_velocity_temp2=DQ.dq_scalar_mult(self.origin1_velocity_temp1,2)
        #генерим ДК обратный для origin1, для этого нужно взять его conjugate и поделить на его же норму
        # inverse_dq_test=self.origin1.dq_inverse()


        # self.frame1_dq_velocity=copy.deepcopy(self.origin1_velocity)
        self.frame1_dq=copy.deepcopy(self.origin1)

        #проверяем вокруг чего вращается СК на конце полученного звена - переводим кватернион в ост и угол (в результатах какая-то лажа)
        axis_for_check, angle_for_check = self.frame1_dq.m_real.q_to_axisangle()
        print(f'Euler axis={axis_for_check} \n Angle={angle_for_check}')

        #подставляем в формулы численные значения углов и скоростей и ускорений
        _coord=Function(f'Q{self.N}')(t)
        Tetta_ = np.radians(float(Tetta))
        V_Tetta_ = np.radians(float(V_Tetta))
        self.origin1.insert_numbers([(_coord,Tetta_)])
        # _vel=Derivative(Function(f'Q{self.N}')(t),t)
        # self.origin1_velocity.insert_numbers([(_vel, V_Tetta_),(_coord,Tetta_)])
        # if (abs(V_Tetta_)<0.0000001):
        #     self.origin1_velocity.m_real_=spQ(0.,0.,0.,0.,)

        #из полученных frame0_dq и frame1_dq нужно извлечь координаты x y z rx ry rz
        #попробуем решить так: попробуем решить так - умножим аналитически 3 кватерниона опследовательных поворотов относительно исходной СК X Y Z - это будем называть углами Эйлера (или самолетным )
        #1) расчет аналитического кватерниона на основе угла и оси
        #2) аналитическое умножение кватернионов
        #3) вывод из получившегося результата углв Эйлера

        # как перевести данные из кватерниона в rx ry rz

        print(f'Link {self.N} completed')

    #попробуем считать аналитически скорость из дуального кватерниона
    def calc_rot_velocity(self, Tetta, V_Tetta):
        f_name=Function(f'Q{self.N}')(t)
        df_name=Derivative(f_name,t)
        ang=symbols('ang')
        vel = symbols('vel')
        #1)поиск производной реальной составляющей ДК (К ориентации - КО)
        diff_DQreal=self.frame1_dq.m_real_.diff()
        #2)начальное положение КО
        q1=self.frame1_dq.m_real_
        # Q1_unit_vector=Matrix(Q1.to_axis_angle()[0])
        q1_vector=Matrix([q1.b, q1.c, q1.d])
        #3) конечное положение КО через время dt
        q2=self.frame1_dq.m_real_+diff_DQreal
        # Q2_unit_vector=Matrix(Q2.to_axis_angle()[0])
        q2_vector = Matrix([q2.b, q2.c, q2.d])
        #4)поиск оси вращения через векторное произведение единичных вектров кватернионов Q1 и Q2
        rot_axis = q1_vector.cross(q2_vector)
        x_axis=rot_axis[0].subs([(df_name,V_Tetta),(f_name,Tetta)])
        y_axis = rot_axis[1].subs([(df_name, V_Tetta), (f_name, Tetta)])
        z_axis = rot_axis[2].subs([(df_name, V_Tetta), (f_name, Tetta)])
        plt.plot(x_axis, y_axis, z_axis, lw=2, c='black')
        # rot_axis=simplify(rot_axis)
        #5) поиск угла поворота между векторами Q1 и Q2 через скалярное произведдение
        ang_velocity=acos(q1_vector.dot(q2_vector)/(q1_vector.norm()*q2_vector.norm()))
        ang_velocity=simplify(ang_velocity)
        #6) создание кватерниона Q3 из полученного угла ang_velocity и оси rot_axis
        q3=spQ.from_axis_angle(rot_axis, ang_velocity)
        #7) поиск производной от текущего угла текущего кватерниона
        q4=spQ.from_axis_angle(self.frame1_dq.m_real_.to_axis_angle()[0],diff_DQreal.a)
        #8) поиск результирующего кватерниона, который содержит информацию о скорости и оси вращения в текущий момент
        q5=q3.mul(q4)
        #9) для упрощения заменяем символы на ang - угол и vel - угловая скорость
        q6=q5.subs([(df_name,vel),
                    (f_name,ang)])
        # print(q6)
        #10) подставляем численные значения в кватернион выше
        q7=q6.subs([(vel,np.radians(V_Tetta)),
                    (ang,np.radians(Tetta))])
        print(q7)

    #методы для получения полностью аналитических формул
    # def get_origin0_real(self):
    #     return nsimplify(self.origin0_dq_real_formula)
    #
    # def get_origin1_real(self):
    #     return nsimplify(self.origin1_dq_real_formula, tolerance=1e-12)
    #
    # def get_origin0_dual(self):
    #     return nsimplify(self.origin0_dq_dual_formula)
    #
    # def get_origin1_dual(self):
    #     return nsimplify(self.origin1_dq_dual_formula)

    #методы для получения аналитических формул с подставленными в них константными числовыми значениями
    #если угол поворота привода tetta не задан, то считается аналитическая формула, если задан - то он сразу подставляется в формулу
    # def get_analytical_dq(self, Tetta=None):
    #     real=self.get_origin1_real().subs([(f'A{self.N}',np.radians(self.alfa)),
    #                                        (f'Q_const{self.N}',np.radians(self.Tetta_const)),
    #                                        (f'd{self.N}',self.d),
    #                                        (f'a{self.N}',self.a)])
    #     dual = self.get_origin1_dual().subs([(f'A{self.N}', np.radians(self.alfa)),
    #                                          (f'Q_const{self.N}', np.radians(self.Tetta_const)),
    #                                          (f'd{self.N}', self.d),
    #                                          (f'a{self.N}', self.a)])
    #     self.formula={}
    #     self.formula['real']=real
    #     self.formula['dual'] = dual
    #     self.analytical_dq=DQ(Qreal_=self.formula['real'], Qdual_=self.formula['dual'])
    #     if not(Tetta is None):
    #         real=real.subs([(f'Q{self.N}',np.radians(Tetta))])
    #         dual = dual.subs([(f'Q{self.N}', np.radians(Tetta))])
    #         Q_real = Q(w=float(real.a), x=float(real.b), y=float(real.c), z=float(real.d))
    #         Q_dual = Q(w=float(dual.a), x=float(dual.b), y=float(dual.c), z=float(dual.d))
    #         self.analytical_dq = DQ(Qreal=Q_real, Qdual=Q_dual)
    #
    #     return self.analytical_dq

    # def get_matrix(self,smplfy=True):
    #     if smplfy:
    #         return nsimplify(simplify(self.analytical_dq.dq_to_matrix_()),tolerance=1e-10)
    #     else:
    #         return self.analytical_dq.dq_to_matrix_()



