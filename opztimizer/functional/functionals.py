from functional.numerical_integrator import quadrature,scalar_product


class Jfunctional():
    def __init__(self,g,S,t,y0,u0):
        self.y0=y0
        self.u0=u0
        self.g=g
        self.y=0
        self.u=0
        self.t=t
        self.S=S
        self.yt1=0
        self.value=self.evaluate(self.y0,self.u0)
    def evaluate(self,y,u):
        self.yt1=y[-1]
        j1=quadrature(y,u,self.t,self.g)
        j2=self.S(self.yt1)
        return j1+j2
    def y_update(self,y):
        self.y=y
        self.yt1=y[-1]
    def control_update(self,u):
        self.u=u
    def update(self,y,u,J_value):
        self.control_update(u)
        self.y_update(y)
        self.value=J_value



def gradient(partial_f,partial_g,y,u,t,p):
    j1f=partial_f(y,u,t)*p
    j2g=partial_g(y,u,t)
    return j1f+j2g

class Gradient():
    def __init__(self,partial_f,partial_g):
        self.partial_f=partial_f
        self.partial_g=partial_g
        self.value=0
    def __call__(self,y,u,t,p):
        self.value=gradient(self.partial_f,self.partial_g,y,u,t,p)
        return self.value
            
