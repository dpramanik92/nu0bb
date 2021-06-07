import numpy as np
import matplotlib.pyplot as plt
import progressbar
import timeit
from tqdm import *

def mbb(X,hier='NH'):
    s12 = (np.sin(X[0]))**2.0
    c12 = 1-s12

    s13 = (np.sin(X[1]))**2.0
    c13 = 1-s13

    ml = X[2]

    dsol = X[3]*1e-5
    datm = X[4]*1e-3

    alp = X[5]
    bet = X[6]

    if(hier=='NH'):
        m1 = ml
        m2 = np.sqrt(ml**2.0+dsol)
        m3 = np.sqrt(ml**2.0+dsol+datm)
    if(hier=='IH'):
        m3 = ml
        m1 = np.sqrt(ml**2.0+datm)
        m2 = np.sqrt(ml**2.0+datm+dsol)

    mbb_r = m1*c12*c13+m2*s12*c13*np.cos(2.0*alp)+m3*s13*np.cos(2.0*bet)
    mbb_i = m2*s12*c13*np.sin(2.0*alp)+m3*s13*np.sin(2.0*bet)

    mbbeta = np.sqrt(mbb_r*mbb_r+mbb_i*mbb_i)

    return mbbeta


def thalf(X,hier='NH',nucl='Xe'):
    gA = 1.27

    if(nucl=='Xe'):
        G = 14.6e-15
    if(nucl=='Ge'):
        G = 2.34e-15
    if(nucl=='Te'):
        G = 14.1e-15

    mBB = mbb(X,hier)
    me = 0.511e6
    Mnu = X[7]

    Thalf = 1.0/G*np.square(me/(Mnu*mBB))

    return Thalf

def edf(x,dist):
    j=0
    for i in dist:
        if(i<=x):
            j=j+1

    return float(j)/float(len(dist))

def signal(X,hier='NH',nucl='Xe',Exposure=1000):

    if(nucl=='Xe'):
        miso = 136
    if(nucl=='Ge'):
        miso = 76
    if(nucl=='Te'):
        miso = 130

    NA = 6.023e23

    t_hlf = thalf(X,hier,nucl)

    sig = np.log(2)*NA*1e3*Exposure/(miso*t_hlf)
    return sig


def Nevents(X,hier,nucl,Exposure,bkg):
      sig = signal(X,hier,nucl,Exposure)

      return sig + bkg


def expected_events(_mbb,Mnu,nucl,Exposure,bkg):
    gA = 1.27

    if(nucl=='Xe'):
        G = 14.6e-15

    if(nucl=='Ge'):
        G = 2.34e-15
    if(nucl=='Te'):
        G = 14.1e-15

    me = 0.511e6


    Thalf = 1.0/G*np.square(me/(Mnu*_mbb))

    if(nucl=='Xe'):
        miso = 136
    if(nucl=='Ge'):
        miso = 76
    if(nucl=='Te'):
        miso = 130

    NA = 6.02214e23


    sig = np.log(2)*NA*1e3*Exposure/(miso*Thalf)
    return sig + bkg



def poiss_likelihood(n_tr,n_te):

    if(n_tr>0):
        p = 2*(n_te-n_tr + n_tr*np.log(n_tr/n_te))
    else:
        p = 0

    return p

class Statistics:
    def __init__(self,_nucl,_mnu,_exp,_bkg):
        self.nucl=_nucl
        self.exp = _exp
        self.bkg = _bkg
        self.mnu = _mnu



    def Calculate_Ntest(self):

        th12 = np.arange(31,36,0.1)
        th13 = np.array([8.5])
        ml = np.array([0.0001])
        dsol = np.array([7.5])
        datm = np.arange(2.4,2.6,0.01)
        alp = np.arange(0,np.pi,0.1)
        bet = np.arange(0,np.pi,0.1)

        self.N_nh = []
        self.N_ih = []

        print('Calculating test events...')

        for t12 in tqdm(th12):
            for t13 in th13:
                for m in ml:
                    for sol in dsol:
                        for atm in datm:
                            for al in alp:
                                for be in bet:
                                    test_params = np.array([t12*np.pi/180.0,
                                        t13*np.pi/180.0,m,sol,atm,al,be,self.mnu])

                                    self.N_nh = Nevents(test_params,'NH',self.nucl,self.exp,self.bkg)
                                    self.N_ih = Nevents(test_params,'IH',self.nucl,self.exp,self.bkg)

        self.N_nh = np.array(self.N_nh)
        self.N_ih = np.array(self.N_ih)


    def chi_sq_min(self,n_tr,n_tes):
        chi = np.min(poiss_likelihood(n_tr,n_tes))
        return chi

    def Test_statistics(self,n_tr):
        chi_min_nh = self.chi_sq_min(n_tr,self.N_nh)
        chi_min_ih = self.chi_sq_min(n_tr,self.N_ih)

        T = chi_min_ih - chi_min_nh

        return T



    def Calculate_ntrue(self,X,hier):
        n_tr = Nevents(X,hier,self.nucl,self.exp,self.bkg)
        print('The numebr of expected events for ',hier,' = ',n_tr)
        return n_tr



def Test_signal(n_tr,n_h0,n_h1):
    chi_0 = poiss_likelihood(n_tr,n_h0)
    chi_1 = poiss_likelihood(n_tr,n_h1)

    return chi_0-chi_1


def sensitivity(mbb):
    nevents = expected_events(mbb,4.8,'Xe',8705,3.8)

    N_dist = np.random.poisson(nevents,1000000)
    N_dist_0 = np.random.poisson(3.4,1000000)

    N_0 = 3.4

    T1 = []
    T2 = []

    for n_tr in N_dist:
        T1.append(Test_signal(n_tr,nevents,N_0))

    T1 = np.array(T1)

    for n_tr in N_dist_0:
        T2.append(Test_signal(n_tr,nevents,N_0))

    T2 = np.array(T2)

    med = np.median(T2)

    sens = edf(med,T1)

    return sens*100


################################################################################

        ## CALCULATION
################################################################################