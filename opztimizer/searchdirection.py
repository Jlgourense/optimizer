from abc import ABC, abstractmethod
import numpy as np
from functional.functionals import Gradient
import logging
class SearchDirection(ABC):
    def __init__(self):
        self.u=0
        self.y=0
        self.p=0
        self.t=0
        self.direction=0
        super().__init__()
    @abstractmethod
    def direction_search(self,u,y,p,t):
        pass


class DeepestDescent(SearchDirection):
    def __init__(self,partial_f,partial_g):
        self.partial_f=partial_f
        self.partial_g=partial_g
        self.grad_J=Gradient(partial_f,partial_g)
        super().__init__()
    def direction_search(self,y,u,t,p):
        logging.info("Calculating direction Deepest descent")
        self.direction=-self.grad_J(y,u,t,p)
        return self.direction
