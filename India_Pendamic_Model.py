import pandas as pd
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

db=pd.read_excel('Cummulative_Data.xlsx', index_col=None)
db1=pd.read_excel('Doubling_Data.xlsx', index_col=None)
# Initial Values
N = 1
no = 1350000000
T = 0.175#0.175 # transmisivity 1.125 without lock down, 0.1 with lockdown
R = 0.04 # Recovery rate
maxT = 3000 # number of days
maxT_display = 50

# function that returns dy/dt
def model(y,t):
    global T
    global R
    global no
    h = y[0]
    i = y[1]
    r = y[2]
    dHdt = -(T*h*i)
    dIdt = (T*h*i)-(R*i)
    dRdt = (R*i)
    return [dHdt,dIdt,dRdt]

# initial condition
Is = 25
Istart = Is/no
Hstart = N-Istart
Rstart = 0
y0 = [Hstart,Istart,Rstart]

# time points
t = np.linspace(0,maxT, 100000)

# solve ODE
y = no*odeint(model,y0,t)
maxT_display_t_id= int(np.argwhere(t>maxT_display)[0])
maxI = int(np.amax(y[:,1]*0.2))
TI = int((y[maxT_display_t_id,2]-y[maxT_display_t_id,0]))
TD = int(y[maxT_display_t_id,2]*0.03)
maxDay = int(t[np.argmax(y[:,1])])
zeroDay = int(t[np.amin(np.where((y[:,1])< (1/no)))])
endI = int(y[maxT_display_t_id,1]*0.2)

print('\nTotal Population',f"{no:,}",'\n')
print('Infected People at Start are',f"{Is:,}",'\n')
print('With Transimtion rate',T,'and Recovery rate',R,'\n')
print('Maximum Infected detected People',f"{maxI:,}",'on',maxDay,'Day','\n')
print('Total Infected',f"{TI:,}",'(',int(TI*100/no),'%)' ,'\n')
print('Total Death',f"{TD:,}",'\n')
print(f"{endI:,}",'Infected People on',maxT_display,'Days','\n')
print('Zero Infected People on',zeroDay,'Days','\n')

# plot results
#plt.plot(db1['Day'],db1['3Day'], label='double in 3 Days')
#plt.plot(db1['Day'],db1['1Week'], label='double in 1 Week')
#plt.plot(db1['Day'],db1['2Week'], label='double in 2 Week')
#plt.plot(db1['Day'],db1['1Month'], label='double in 1 Months')
#plt.plot(db['Day'],db['I'], label='Infected Recored Data')
#plt.plot(db['Day'],db['R'], label='Recovered Recored Data')
plt.plot(db['Day'],db['I_Increase'], label='Daily Confirm Cases')
plt.plot(db['Day'],db['R_Increase'], label='Daily Removed (Recovered+Death) Cases')
plt.plot(db['Day'],db['I-R_Increase'], label='Daily Active Cases')
#plt.plot(t,y[:,0], label='Healthy')
#plt.plot(t,y[:,1], label='Infected')
#plt.plot(t,y[:,1]*0.2, label='Infected Detected')
#plt.plot(t,y[:,2], label='Recovered')
#plt.plot(t,y[:,2]*0.03, label='Death')
#plt.xlabel('Days')
#plt.ylabel('Normalized Number of People')
#plt.ylim(10,15000)
#plt.xlim(10,maxT_display)
plt.legend(loc = 'upper left')
#plt.yscale("log")

#plt.set_label()
#plt.legend(label=['Healthy', 'Infected','Recovered'])
plt.show()