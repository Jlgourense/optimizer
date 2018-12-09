import numpy as np
import logging
def quadrature(y,u,t,function):
    n=len(t)-1
    t0=t[0]
    t1=t[n]
    f=function
    dt=(t1-t0)/(n)
    logging.info("t0 t1 n dt {} {} {} {}".format(t0,t1,n,dt))
    fsum=(dt/2)*(f(y[0],u[0],t[0])+f(y[n],u[n],t[n]))
    fsum=fsum+np.sum(f(y[1:(n)],u[1:(n)],t[1:(n)]))*dt
    return fsum
    
def scalar_product(w,v,t):
    dt=t[1]-t[0]
    n=len(t)-1
    f=w*v
    fsum=(dt/2)*(f[0]+f[n])
    fsum=fsum+np.sum(f[1:n])*dt
    return fsum
    
