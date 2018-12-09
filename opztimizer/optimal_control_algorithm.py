import numpy as np
from searchdirection import DeepestDescent
from searchline import ArmijoSearchLine
from functional.functionals import Jfunctional
from system_solvers.solvers import StateSystem,AdjointSystem
import logging

####test system####
f=lambda y,u,t:y+u

s=lambda yp:yp**2
uexacta=lambda t:-((2*np.e**2)/(1+np.e**2))*np.e**(-t)
yexacta=lambda t:((1)/(1+np.e**2))*np.e**(t)+((np.e**2)/(1+np.e**2))*np.e**(-t)
pexacta=lambda t: ((4*np.e**2)/(1+np.e**2))*np.e**(-t)
uf=lambda t:3*np.sin(4*t)+uexacta(t)
partial_f_u=lambda y,u,t:1
partial_g_u=lambda y,u,t:2*u
partial_f_y=lambda y,u,t:1
partial_g_y=lambda y,u,t:0
sprime=lambda y:2*y
g=lambda y,u,t:u**2
time_interval=[0,1]
N=100
t=np.linspace(0,1,N+1)
u=uf(t)
####



class DescentAlgorithm():
    def __init__(self,u_f,f,g,s,sprime,partial_f_u,partial_g_u,partial_f_y,partial_g_y,N=100,epsilon=0.001,time_interval=[0,1]):
        self.y=0
        self.set_solutions=[]
        self.grad_J=1
        self.direction=0
        self.epsilon=epsilon
        self.convergence=False
        self.f=f
        self.g=g
        self.s=s
        self.sprime=sprime
        self.t=t
        self.partial_f_u=partial_f_u
        self.partial_g_u=partial_g_u
        self.partial_f_y=partial_f_y
        self.partial_g_y=partial_g_y
        self.f_p=lambda y,u,t,p:self.partial_f_y(y,u,t)*p

        logging.info("Creating direction searcher")
        self.direction_searcher=DeepestDescent(self.partial_f_u,partial_g_u)
        logging.info("Creating line searcher")
        self.line_searcher=ArmijoSearchLine()

        
        ###initial conditions
        self.y0=1
        self.p0=1
        ###solver parameters 
        self.N=N
        self.time_interval=time_interval
        self.t0=self.time_interval[0]
        self.t1=self.time_interval[1]
        self.t=np.linspace(self.t0,self.t1,self.N+1)
        #creating u
        self.u=u_f(self.t)
        args_state=(self.N,self.time_interval,self.f)
        
        args_adjoint=(self.partial_f_y,self.partial_g_y,self.N,self.time_interval,self.f_p)
        
        logging.info("Creating solvers")
        self.state_system=StateSystem(*args_state)
        self.adjoint_system=AdjointSystem(*args_adjoint)
        
        
        
        logging.info("Creating Functional and solving first y and p")
        self.y=self.state_system.y_calculation(self.u,self.y0)
        
        logging.info ("creating funciontal with y {}...{} u {} ...{}".format(self.y[0],self.y[-1],self.u[0],self.u[-1]))
        self.J=Jfunctional(self.g,self.s,self.t,self.y,self.u)

        super().__init__()
    def update(self,uk,yk,Jk_value):
        self._last_values=np.array([[self.u],[self.y],[self.J]])
        self.u=np.array(uk)
        self.y=np.array(yk)
        self.J.update(self.y,self.u,Jk_value)
        self._new_values=np.array([[self.u],[self.y],[self.J]])
        logging.info("MAIN Updating u y J {} {} {}".format(self.u,self.y,self.J.value))

    def line_search(self):
        [uk,yk,Jk_value]=self.line_searcher.line_search(self.u,self.y,self.t,self.J,self.direction,self.state_system,self.y0)
        self.update(uk,yk,Jk_value)
    def direction_search(self):
        k=0
        while (not self.check_convergence()):
            logging.info("MAIN direction search time step k {}".format(k))
            self.p0=self.sprime(self.y[-1])
            logging.info("p0 {}".format(self.p0))
            self.p=self.adjoint_system.y_calculation(self.u,self.y,self.p0)
            self.direction=self.direction_searcher.direction_search(self.y,self.u,self.t,self.p)
            self.grad_J=-self.direction
            logging.info(" Direction {}{}....{}".format(self.direction[0],self.direction[1],self.direction[-1]))
            self.line_search()
            k=k+1
            if ((k%5)==0):
                self.set_solutions.append([self.y,self.u,self.p])
    def check_convergence(self):
        logging.info("Checking convergence")
        grad_norm=np.linalg.norm(self.grad_J)
        if grad_norm<self.epsilon:
            self.convergence=True
        logging.info("grad_norm {}".format(grad_norm))
        return self.convergence
