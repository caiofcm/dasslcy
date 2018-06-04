####################################################################################################
#                                                                                                  #
#                       This is a template for the dasslc module in python2.                       #
#                       Use it and modify at free will.                                            #
#                                                                                                  #
#                       Author: ataide@peq.coppe.ufrj.br                                           #
#                       Version: 0.1                                                               #
#                       Last modification: Apr 26, 2018 @ 15:01:30                                 #
#                                                                                                  #
####################################################################################################

          #--out--#                 #-- req. inp. --#  #----- optional inputs -----#
### call: t,  y, yp = dasslc.solve (resfun, tspan, y0, yp0, rpar, rtol, atol,  index)

## Import the modules
import dasslc
import numpy as np
import matplotlib.pyplot as plt


#################################### Defining the residual functions ###############################

def model0(t,y,yp): #--------------- Minimum of 3 input arguments
    res = np.empty(1) #------------- Always allocate res as a numpy array, even if it has len = 1.
    res[0] = yp[0] + 2*y[0] #------- Declare the residual  
    ires = 0 #---------------------- Set ires = 0 if everything is ok
    return res, ires #-------------- Beware: ires must always be returned as second output.


def model1(t,y,yp): #--------------- Just another example
    res = np.empty(2)
    res[0] = yp[0]-20*np.cos(20*t)
    res[1] = yp[1]+20*np.sin(20*t)
    return res, 0 #----------------- ires can be a literal


def model2(t,y,yp,par): #------------- Maximum of 4 input arguments
    res = np.empty(3)         
         
    k1 = par[0]                     #||||||||||||||||||||||||||||||||||||||||||||||||
    k2 = par[1]                     #|                                              |
    Ca = y[0]; dCa = yp[0] #--------#| aliasing is optional, but always encoraged   |
    Cb = y[1]; dCb = yp[1]          #|                                              |
    Cc = y[2]; dCc = yp[2]          #||||||||||||||||||||||||||||||||||||||||||||||||

    res[0] = -k1*Ca - dCa  
    res[1] = k1*Ca - k2*Cb - dCb
    res[2] = k2*Cb - dCc
    ires = 0
    return res, ires


def model3(t,y,yp,par): #----------- The parameter may be a whole class
    res = np.empty(5)
    res[0] = yp[0] - y[2]
    res[1] = yp[1] - y[3]
    res[2] = yp[2] + y[4]*y[0]
    res[3] = yp[3] + y[4]*y[1] + par.g
    if (par.dae == 3):
        res[4] = y[0]*y[0] + y[1]*y[1] - par.L*par.L
        ires = 0
    elif (par.dae == 2):
        res[4] = y[0]*y[2] + y[1]*y[3]
        ires = 0
    elif (par.dae == 1):
        res[4] = y[2]**2 + y[3]**2 - par.g*y[1] - par.L**2*y[4]
        ires = 0
    elif (par.dae == 0):
        res[4] = yp[4] + 3*y[3]*par.g/par.L**2
        ires = 0
    else:
        print "Invalid index."
        ires = -1

    return res, ires


######################################## Solve model0 ##############################################

t0 = np.array([0,1]) #---------------------- Integration interval with initial and final time
y0 = np.array([1])   #---------------------- Initial condition

t, y, yp = dasslc.solve(model0,t0,y0) #---#| The simplest call to dasslc,
                                          #| with all the mandatory inputs and outputs.
                                          #| y and yp are equally spaced in all time span
# Plot results
plt.figure(1)
plt.subplot(211)
plt.plot(t,y)
plt.ylabel('y')
plt.title('Model0 Solution')
plt.subplot(212)
plt.plot(t,yp)
plt.xlabel('time')
plt.ylabel('yp')


######################################### Solve model1 #############################################

t0 = np.linspace(0,1,100) #----------------#| The time span can also be a vector.
                                           #| In this case y and yp are returned at all values of t

y0 = np.array([0,1])      #------------------ Initial condition
yp0 = np.array([1,0])     #------------------ Derivatives at initial condition (optional)

t, y, yp = dasslc.solve(model1,t0,y0,yp0) #-- Call with the optional yp0


# Plot results
plt.figure(2)
l1, l2 = plt.plot(t,y)
plt.ylabel('y')
plt.xlabel('time')
plt.title('Model1 Solution')
plt.legend([l1,l2],["y1","y2"])

######################################### Solve model2 #############################################

t0 = np.array([500]) #------------#| You can also specify only the final time.
                                  #| In this case y and yp are equally spaced in [0 t0]
y0 = np.array([1,0,0])
yp0 = None #----------------------#| If you are not passing an optional input
                                  #| but is passing the next one, define it as None 

par = np.array([0.01,0.02]) #------- The optional parameter vector
atol = 1e-8 #----------------------- The absolute tolerance
rtol = 1e-6 #----------------------- The relative tolerance

t, y, yp = dasslc.solve(model2,t0,y0,yp0,par,rtol,atol) # Call with optional arguments (yp0 = None)

# Plot results
plt.figure(3)
l1, l2, l3 = plt.plot(t,y)
plt.ylabel('y')
plt.xlabel('time')
plt.title('Model2 Solution')
plt.legend([l1,l2,l3],["Ca","Cb","Cc"])


######################################### Solve model3 #############################################

class pend_par: #----------------#|
    g = 9.81                     #| Defining the parameter class for 
    L = 1.0                      #| the pendulum model
    dae = 3                      #|

t0 = np.linspace(0,50,10000)
y0 = np.array([1,0,0,0,0])
yp0 = None
par = pend_par() #----------------- The optional parameter class initialization
atol = 1e-10
rtol = 1e-8
index = np.array([1,1,2,2,3]) #---- The dependent variable index vector (needed for high index DAE)

t, y, yp = dasslc.solve(model3,t0,y0,yp0,par,rtol,atol,index)

# Plot results
plt.figure(4)
l1, l2, l3, l4, l5 = plt.plot(t,y)
plt.ylabel('y')
plt.xlabel('time')
plt.title('Model3 Solution')
plt.legend([l1,l2,l3,l4,l5],["x","y","vx","vy","mu"])

## Show all figures
plt.show()

