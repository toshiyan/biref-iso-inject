
# general
import numpy as np

# cmblensplus/utils/
import constant as c

# physical units
Tcmb  = 2.726e6       # CMB temperature
ac2rad = np.pi/10800. # arcmin -> rad
eV2Mpc = 806554.815354305*3.0856775814914e22*(2*np.pi)

# cosmological parameters (Planck 2018, arXiv:1807.06209, TT+TE+EE+lowE+lensing)
H0    = 67.36
Om    = 0.3153
As    = 2.1e-09
ns    = 0.9649
ombh2 = 0.02237
zcmb  = 1059.94
zeq   = 3402

# derived cosmological parameters
acmb  = 1./(1.+zcmb)
aeq   = 1./(1.+zeq)
h0    = H0/100.
H0Mpc = H0/c.C
Ob    = ombh2/h0**2
Or    = Om*aeq
Ov    = 1-Om # flat universe
omch2 = Om*h0**2 - ombh2
cps   = {'H0':H0,'Om':Om,'Ov':Ov,'w0':-1,'wa':0.}


# phi parameters
def params_phi(model='ALP'):
    '''
    n      : n=1 for ALP and n=2 for EDE potential
    tf     : f_a/phi_ini; the symmetry breaking scale normalized by phi_ini, dimensionless
    '''
    if model=='ALP':
        n, tf = 1, 100.
    if model=='EDE':
        n, tf = 2, np.sqrt(8*np.pi)
    return n, tf


def Ea(a,rad=1.): # H(a)/H0
    return np.sqrt( Om/a**3 + Ov + rad*Or/a**4 )


def dlnEada(a):
    return -(1.5*a*Om+2*Or)/(a**5*Ea(a)**2)



def V_phi(x,logm,tf,n,deriv=0):
    '''
    x      : pseudoscalar field normalized by phi_ini, dimensionless
    logm   : logarithm of the mass parameter in the unit of eV
    tf     : the symmetry breaking scale normalized by phi_ini, dimensionless
    deriv  : the order of derivative with respect to phi
    '''
    
    m = 10**(logm)*eV2Mpc
    
    if deriv == 0: # potential
        return m**2*tf**2*(1.-np.cos(x/tf))**n

    if deriv == 1: # dV/dphi
        return n*m**2*tf*(1.-np.cos(x/tf))**(n-1)*np.sin(x/tf)

    if deriv == 2: # d^2V/dphi^2
        if n == 1:
            return m**2*np.cos(x/tf)
        else:
            return n*m**2*(1.-np.cos(x/tf))**(n-2)*(-1.+np.cos(x/tf)+n*np.sin(x/tf)**2)


def EoM_phi(a, x, logm, tf, n, xi=None, eps=1e-3, z_start=1e2,z_end=10):
    
    '''
    # Solving the following equations
    #   dx0/da = x1
    #   dx1/da = d^2x0/da^2 = - ( 4/a + dlnH/da ) dx0/da + (dV/dphi)/(a^2H^2)
    # where x0 is varphi=phi/phi_ini and x1 is d(varphi)/da. 
    # If V = m^2 x0^2 / 2, the above second equation becomes
    #   dx1/da = d^2x0/da^2 = - ( 4/a + dlnH/da ) dx0/da + (m/H0)^2 x0 / (a^2E^2)
    '''

    # conformal expansion rate
    calH = a*H0Mpc*Ea(a)

    # mass
    m = 10**(logm)*eV2Mpc

    # redshift
    z = 1./a - 1.
    
    if xi is not None and z>z_end and z<z_start:
        # Energy injection to ALPs
        # Regularized source term 
        # For |pi| >> eps, pi*S ~ Q 
        # Src = Q/(dphi/deta)/phi_ini^2 
        #     = (xi calH rho_m)/(a calH dphi/da)/phi_ini^2
        #     = (xi/a)(m^2/2)(rho_m0/rho_phi_ini)/(dphi/da)
        frac = 1e-3 # energy density fraction of ALP to matter
        Q    = (xi/a) * (0.5*m**2/frac/a**3)
        Src  = Q * x[1] / (x[1]**2 + eps**2)
        #print(x[1])
    else:
        Src = 0.
    
    return [ x[1], - ( (4./a+dlnEada(a))*x[1] + ( V_phi(x[0],logm,tf,n,deriv=1) - Src ) / calH**2 ) ]


# Define the potential and its derivative
def V_phi_quad(m,phi):
    return m**2 * phi**2 / 2

def dV_dphi_quad(m,phi):
    return m**2 * phi
    
# ALP as spectator
def equations_nb(t, y, m):
    phi, dphi, a = y
    rho_phi = dphi**2/2. + V(m,phi)
    H = H0Mpc * np.sqrt( Om/a**3 + Or/a**4 + 1-Om) 
    ddphi = -3 * H * dphi - dV_dphi(m,phi)
    da = H * a
    return [dphi, ddphi, da]


