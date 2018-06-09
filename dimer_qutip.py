import numpy as np
from math import *
from functools import partial

from qutip import *



si = qeye(2)
sx = sigmax()
sy = sigmay()
sz = sigmaz()
'''
		operadores
'''
def n2_operator():
	return tensor(si,si)+0.5*(tensor(sz,si)+tensor(si,sz))
	
def n1_operator():
	return tensor(si,si)-0.5*(tensor(sz,si)+tensor(si,sz))

def hopping_operator():
	return (tensor(sx,si)-tensor(si,sx))

def repulsionee_operator():
	return 0.5*(tensor(si,si)+tensor(sz,sz))

def dimer_Hamiltonian_free(J,U):
	H_hopping_Op = -J* hopping_operator()
	H_coulomb_Op = U * repulsionee_operator()
	return H_hopping_Op + H_coulomb_Op

'''
	
		THERMAL STATE

'''
def thermal_state_qutip(Ham, beta):
	bH = -beta * Ham
	expbH = bH.expm()
	Z = expbH.tr()
	rho_thermal = expbH / Z
	return rho_thermal  


'''

	EXTERNAL POTENTIAL

'''
'''
	funcao senoidal
'''	
def Vext_sin(dict_args, t, *args):
	A0 = dict_args['A0']
	At = dict_args['At']
	omega = dict_args['omega']
	tau = dict_args['tau']
	return A0 + At * sin(omega * t / tau)	



import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-J','--J', type=float, default=-1.0, help='hopping parameter')
parser.add_argument('-U','--U', type=float, default=2.0, help='Coulomb potential')

# thermo arguments 
parser.add_argument('-beta','--beta', type=float, default=100.0, help='beta = 1 / kT. Default is 100 ~ ground state')


# time-dependent arguments
parser.add_argument('-dt','--dt', type=float, default=0.05, help='Step of time to process evolution')
parser.add_argument('-nstepsQuTip','--nstepsQuTip', type=int, default=5000, help='Step of time to process evolution')
parser.add_argument('-tau','--tau', type=float, default=10.0, help='Minimum value of tau for time evolution')

# external potentials
parser.add_argument('-A0_1','--A0_1', type=float, default=0.5, help='Constant external potential in site 1')
parser.add_argument('-A0_2','--A0_2', type=float, default=-0.5, help='Constant external potential in site 2')

parser.add_argument('-At_1','--At_1', type=float, default=7.0, help='Time-dependent coefficient for external potential in site 1')
parser.add_argument('-At_2','--At_2', type=float, default=-7.0, help='Time-dependent coefficient for external potential in site 2')


parser.add_argument('-omega1','--omega1', type=float, default=0.5, help='Time-dependent oscillation frequency for external potential in site 1 (units of pi)')
parser.add_argument('-omega2','--omega2', type=float, default=0.5, help='Time-dependent oscillation frequency for external potential in site 2 (units of pi)')


parser.add_argument('-show','--show', type=int, default=1, help='Flag for plot')
parser.add_argument('-showani', '--show_animation', type=int, default=0, help="Flag to see result animated or no")


opts = parser.parse_args()
'''
		free model parameters
'''
J = opts.J
U = opts.U

'''
	parametros do potencial
'''
omega1 = opts.omega1 * pi 
omega2 = opts.omega2 * pi 

A0_1 = opts.A0_1
At_1 = opts.At_1

A0_2 = opts.A0_2
At_2 = opts.At_2


'''
	temperatura
'''
beta = opts.beta


'''
	tempo para ligar o potencial no maximo
'''
dt = opts.dt
tau = opts.tau
nstepsQuTip = opts.nstepsQuTip
optionsQuTip = qutip.Options(nsteps=nstepsQuTip)

if(dt is None):
	ts = [0, tau]
	nts = 2
else:
	nts = int(tau / dt) + 1
	ts = np.linspace(0, tau, nts, endpoint=True)


# Hamiltonian
Ham_dimer_free_Op = dimer_Hamiltonian_free(J,U)

dim = Ham_dimer_free_Op.shape[0]

# number operator
n1_Op = n1_operator()
n2_Op = n2_operator()


dict_potentials = [{'A0': A0_1, 'At': At_1, 'omega': omega1, 'tau': tau}, {'A0': A0_2, 'At': At_2, 'omega': omega2, 'tau': tau}]


'''
	hamiltoniano inicial
'''
Ham_initial = Ham_dimer_free_Op 
Ham_initial+= n1_Op * partial(Vext_sin, dict_potentials[0])(0)
Ham_initial+= n2_Op * partial(Vext_sin, dict_potentials[1])(0)

rho0 = thermal_state_qutip(Ham_initial, beta)

'''
	populacoes iniciais
'''
ergs_initial, n_eigstates_initial = qutip.Qobj(Ham_initial).eigenstates(sort="low")
p_n_initial = np.zeros((dim,1))
for n in range(dim):
    p_n_initial[n] = qutip.expect(rho0, n_eigstates_initial[n]) 


Ht = [Ham_initial, [n1_Op, partial(Vext_sin, dict_potentials[0])], [n2_Op, partial(Vext_sin, dict_potentials[1])]]


'''
	estado evoluido em cada instante de tempo
'''
rho_t = []

for ti in range(nts):
	rho_t.append(0.0 * rho0)
	


for n in range(dim):
	# final evolved state
	n_state_initial_q = n_eigstates_initial[n]
	n_state_tau_q = mesolve(Ht, n_state_initial_q, ts,[],[], options=optionsQuTip).states
	
	for ti in range(nts):
		rho_t[ti]+= p_n_initial[n][0] * qutip.ket2dm(n_state_tau_q[ti])
		
'''
	calculando as densidades instantaneas
'''		
n1_inst = np.zeros(nts)
n2_inst = np.zeros(nts)		
		
for ti in range(nts):
	t = ts[ti]		
	n1_inst[ti] = qutip.expect(rho_t[ti], n1_Op)
	n2_inst[ti] = qutip.expect(rho_t[ti], n2_Op)
	

rho_tau = rho_t[-1]

# ou diretamente
rho_tau_2 = mesolve(Ht, rho0, ts,[],[], options=optionsQuTip).states[-1]




import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

cor1 = 'red'
cor2 = 'blue'

nyticks=5
nxticks=5

if(opts.show):
	fig, ax = plt.subplots()

	ax.plot(ts, n1_inst, color=cor1, lw=2.0, label=r'$\langle n_1 (t) \rangle$')
	ax.plot(ts, n2_inst, color=cor2, lw=2.0, label=r'$\langle n_2 (t) \rangle$')
	
	yticks = np.linspace(0, 2, nyticks)
	xticks = np.linspace(0, 1.0*tau, nxticks)
	
	ax.set_yticks(yticks)
	ax.set_yticklabels([r'$%.2f$' % y for y in yticks], fontsize=14)	

	ax.set_xticks(xticks)
	ax.set_xticklabels([r'$%.2f$' % x for x in xticks], fontsize=14, va='center')	
	
	ax.set_xlabel(r'$t$', fontsize=20)
	ax.set_ylabel(r'$ n_j (t) $', fontsize=20)
	
	ax.set_title(r'$U /J = %.2f, k_B T / J= %.2f, \tau \times J = %.2f$' % (U, 1.0 / beta, tau))
	
	ax.yaxis.set_tick_params(which='major', pad=2.5, length=7.5, direction='inout')
	ax.xaxis.set_tick_params(which='major', pad=10.5, length=7.5, direction='inout')
	
	ax.set_ylim(0-0.0125, 2+0.0125)
	
	ax.set_xlim(0, ts.max())
	ax.text(0.9*tau, n1_inst[-1]+0.0125, r'$n_1(t)$', fontsize=20, va='bottom', color=cor1)
	ax.text(0.9*tau, n2_inst[-1]-0.025, r'$n_2(t)$', fontsize=20, va='top', color=cor2)  	

	plt.show()
	

if(opts.show_animation):
	
	
	fig, ax = plt.subplots()
	
	plt.ion()
	
	ti = 0
	l1, = ax.plot(ts[ti], n1_inst[ti], marker='o', ms=3, color=cor1, mec='None')
	l2, = ax.plot(ts[ti], n2_inst[ti], marker='o', ms=3,color=cor2, mec='None')

	minn = min(n1_inst.min(),n2_inst.min())
	maxn = max(n1_inst.max(),n2_inst.max())

	ax.text(0.1, minn-0.0125, r'$n_1(t)$', fontsize=20, va='bottom', color=cor1)
	ax.text(0.1, maxn+0.0105, r'$n_2(t)$', fontsize=20, va='top', color=cor2)  
		
	yticks = np.linspace(float(r"%.2f" % minn), float(r"%.2f" % maxn), nyticks)
	xticks = np.linspace(0, 1.0*tau, nxticks)
	
	ax.set_yticks(yticks)
	ax.set_yticklabels([r'$%.2f$' % y for y in yticks], fontsize=14)	

	ax.set_xticks(xticks)
	ax.set_xticklabels([r'$%.2f$' % x for x in xticks], fontsize=14, va='center')	
	
	ax.set_xlabel(r'$t$', fontsize=20)
	ax.set_ylabel(r'$ n_j (t) $', fontsize=20)
	
	ax.set_title(r'$U /J = %.2f, k_B T / J= %.2f, \tau \times J = %.2f$' % (U, 1.0 / beta, tau))
	
	ax.yaxis.set_tick_params(which='major', pad=2.5, length=7.5, direction='inout')
	ax.xaxis.set_tick_params(which='major', pad=10.5, length=7.5, direction='inout')
	
	ax.set_ylim(minn-0.0125, maxn+0.0125)
	
	ax.set_xlim(0, ts.max())
	
	plt.draw()
	
	for ti in range(1,nts):
		n1 = ax.plot(ts[0:ti], n1_inst[0:ti], color=cor1, marker='o', ms=3, mec='None', linestyle = '-',lw=2.0)
		n2 = ax.plot(ts[0:ti], n2_inst[0:ti], color=cor2, marker='o', ms=3, mec='None', linestyle='-',lw=2.0)

		plt.draw()
		
		plt.pause(0.05)

	
	#ax.text(0.9*tau, n1_inst[-1]+0.0125, r'$n_1(t)$', fontsize=20, va='bottom', color=cor1)
	#ax.text(0.9*tau, n2_inst[-1]-0.025, r'$n_2(t)$', fontsize=20, va='top', color=cor2)  	

	plt.show()	
