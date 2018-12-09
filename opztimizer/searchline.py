from abc import ABC, abstractmethod
import numpy as np
from functional.numerical_integrator import scalar_product
import logging
class SearchLine(ABC):
    def __init__(self):
        self.condition=False
        super().__init__()
    @abstractmethod
    def check_condition(self):
        pass
    @abstractmethod
    def line_search(self,uk,yk,Jk,direction):
        pass




class ArmijoSearchLine(SearchLine):
    def __init__(self,s=0.1,sigma=0.1,beta=0.9):
        self.s=s
        self.sigma=sigma
        self.beta=beta
        self.Jnext_value=0
        self.uknext=0
        self.yknext=0
        self.direction=0
        self.uk=0
        self.yk=0
        self.Jk=0
        self.J=0
        self.grad_J=0
        super().__init__()
    def check_condition(self):
        logging.info("checking condition line search")
        from scipy.integrate import simps
        cond1=self.J.value+self.sigma*scalar_product(self.grad_J,(self.uknext-self.uk),self.t)
        cond=self.Jnext_value<cond1
        self.condition=cond
        if (self.condition):
            logging.info("searchline condition achieved exiting loop")
        return cond
    def calculate_step(self,l,state_system,y0):
        self.uknext=self.uk+(self.beta)**l*self.s*self.direction
        logging.info("norm delta(u) {}".format(np.linalg.norm(self.uknext-self.uk)))
        self.yknext=state_system.y_calculation(self.uknext,y0)
        self.Jnext_value=self.J.evaluate(self.yknext,self.uknext)
        logging.info("calculated step values uknext yknext Jknext J {}..{} {}..{} {} {}".format(self.uknext[0],self.uknext[-1],self.yknext[0],self.yknext[-1],self.Jnext_value,self.J.value))
    def line_search(self,uk,yk,t,J,direction,state_system,y0):
        self.uk=uk
        self.yk=yk
        self.J=J
        self.direction=direction
        self.grad_J=-direction
        self.t=t
        l=1
        while (not self.condition):
            logging.info("calculating step l {}".format(l))
            self.calculate_step(l,state_system,y0)
            self.check_condition()
            l=l+1
            if l>10:
                logging.warning("max iterations reached searchline")
                break
        logging.info("reseting line_search condition")
        self.condition=False
        return [self.uknext,self.yknext,self.Jnext_value]

