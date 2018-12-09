import logging
import numpy as np
def explicit_euler_state(y,u,x,h,y1,function):
    logging.info("time step{}".format(h))
    logging.info("initial condition {}".format(y1))
    f=function
    y[0]=y1
    logging.info("Setting initial condition  {}".format(y[0]))
    for i in range(len(x)-1):
        y[i+1]=y[i]+h*f(y[i],u[i],x[i])
        #y[i+1]=y[i]+h*x[i]**2
        
    return y


def explicit_euler_adjoint(y,y_state,u,x,h,y1,function,partial_g):
    logging.info("time step{}".format(h))
    logging.info("initial condition {}".format(y1))
    y[-1]=y1


    f=function
    y[0]=y1
    logging.info("Setting initial condition y_flipped[0] {}".format(y[0]))
    for i in range(len(x)-1):
        n=len(x)-2-i
        y[n]=y[n+1]+h*f(y_state[n+1],u[n+1],x[n+1],y[n+1])+h*partial_g(y_state[n+1],u[n+1],x[n+1])
        #y[i+1]=y[i]-h*x[i]**2
        
    return y
