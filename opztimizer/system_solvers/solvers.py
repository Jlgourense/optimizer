from abc import ABC, abstractmethod
import numpy as np
from system_solvers.integrators import explicit_euler_adjoint as eea
from system_solvers.integrators import explicit_euler_state as ees



class Ode1Solver(ABC):
    def __init__(self,N,time_interval,function):
        self.y=np.array([])
        self.N=N
        self.t0=time_interval[0]
        self.t1=time_interval[1]
        self.h=(self.t1-self.t0)/N
        self.f=function
        self._y_inizialization()
        self.t=np.linspace(self.t0,self.t1,self.N+1)
        super().__init__()
    def _y_inizialization(self):
        self.y=np.zeros([1,self.N+1])[0]
    @abstractmethod
    def y_calculation(self):
        pass
        

class StateSystem(Ode1Solver):
    def __init__(self,*args):
        super().__init__(*args)
    def y_calculation(self,u,y0):
        self.y=ees(self.y,u,self.t,self.h,y0,self.f)
        return self.y


class AdjointSystem(Ode1Solver):
    def __init__(self,partial_f_y,partial_g_y,*args):
        self.y_state=0
        self.partial_f=partial_f_y
        self.partial_g=partial_g_y
        
        
        super().__init__(*args)
    def y_calculation(self,u_control,y_state,y0):
        self.y=eea(self.y,y_state,u_control,self.t,self.h,y0,self.f,self.partial_g)
        return self.y
    def control_update(self,u):
        self.u=u
    def y_update(self,y_state):
        self.y_state=y_state

