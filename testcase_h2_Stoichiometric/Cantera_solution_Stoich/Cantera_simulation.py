import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import PyCSP.Functions as csp

os.makedirs('figures_Stoichiometric/', exist_ok=True)

#======== CANTERA SIMULATION ========#
#create gas from original mechanism file hydrogen.cti
gas = csp.CanteraCSP('mech/New_h_mech.yaml')

#set the gas state
T = 1000
P = 2*ct.one_atm # match senk.out initial cdts
gas.TP = T, P
# Set constant pressure
gas.constP = True
phi = 1
gas.set_equivalence_ratio(phi, 'H2', 'O2:1, N2:3.7599, AR:0.0001')

#integrate ODE
r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])
states = ct.SolutionArray(gas, extra=['t'])


evals = []
Revec = []
Levec = []
fvec = []
#sim.set_initial_time(0.0)
sim.initial_time = 0.0
while sim.time < 1e-3:
    sim.step()
    states.append(r.thermo.state, t=sim.time)
    print('%10.3e %10.3f %10.3f %14.6e' % (sim.time, r.T, r.thermo.P, r.thermo.u))
    
    lam,R,L,f = gas.get_kernel()
    evals.append(lam)
    Revec.append(R)
    Levec.append(L)
    fvec.append(f)

evals = np.array(evals)
Revec = np.array(Revec)
Levec = np.array(Levec)
fvec = np.array(fvec)




#======== Plot comparison ========#
plt.figure(figsize=(10, 8))

# Plot Temperature
plt.subplot(2, 2, 1)
plt.plot(states.t, states.T,'-', label='Cantera', lw=2)
plt.xlabel('Time (s)')
plt.ylabel('Temperature (K)')
plt.xlim(0, 0.001)
plt.legend()
plt.show() # just to see that results are good 

