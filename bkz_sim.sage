

# This file was *autogenerated* from the file bkz_sim.sage
from sage.all_cmdline import *   # import sage library

from fpylll import *

from fpylll import *
from fpylll import BKZ as BKZ_FPYLLL
from fpylll import LLL as LLL_FPYLLL
from fpylll import GSO, IntegerMatrix, FPLLL, Enumeration, EnumerationError, EvaluatorStrategy
from fpylll.tools.quality import basis_quality
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.util import gaussian_heuristic
from fpylll.tools.bkz_simulator import  simulate, simulate_prob, averaged_simulate_prob

from itertools import chain
import time
from time import perf_counter
from copy import deepcopy

import kknsim

def phione_profile(n,q):
    """
    Outputs profile of phi 1-orthogonal lattice and scaling factor.
    """
    onev = vector( [1 ]*n+[0 ]*n )
    n_ = n-1
    Bq = []
    for i in range(n_):
        qv = vector(i*[0 ]+[q]+(n+n-i-1 )*[0 ])
        qv = qv - onev*qv/(n) * onev
        Bq.append( qv )

    Bq = matrix(Bq)
    denom = Bq.denominator()

    print( f"denom: {denom}" )

    Bq = denom*Bq
    Bq = Bq.change_ring(ZZ)
    B = IntegerMatrix.from_matrix( Bq )
    G = GSO.Mat( B, float_type='mpfr' )
    G.update_gso()

    ldenom = log(denom)  #we assume, the tail has the same shape as qary vectors
    lgq = log(q)
    l = [ G.get_r(i,i)  for i in range(n_) ] + [ G.get_r(i,i)/q**2  for i in range(n_) ]

    return l, denom

gh(b) = lambda b: b/(2*pi*e)
@CachedFunction
def ball_log_vol(n):
    return float((n/2.) * log(pi) - lgamma(n/2. + 1))

gh_constant = {1:0.00000,2:-0.50511,3:-0.46488,4:-0.39100,5:-0.29759,6:-0.24880,7:-0.21970,8:-0.15748,9:-0.14673,10:-0.07541,11:-0.04870,12:-0.01045,13:0.02298,14:0.04212,15:0.07014,16:0.09205,17:0.12004,18:0.14988,19:0.17351,20:0.18659,21:0.20971,22:0.22728,23:0.24951,24:0.26313,25:0.27662,26:0.29430,27:0.31399,28:0.32494,29:0.34796,30:0.36118,31:0.37531,32:0.39056,33:0.39958,34:0.41473,35:0.42560,36:0.44222,37:0.45396,38:0.46275,39:0.47550,40:0.48889,41:0.50009,42:0.51312,43:0.52463,44:0.52903,45:0.53930,46:0.55289,47:0.56343,48:0.57204,49:0.58184,50:0.58852}
def log_gh(d, logvol=0):
    if d < 49:
        return gh_constant[d] + float(logvol)/d

    return 1./d * float(logvol - ball_log_vol(d))

def delta(k):
    assert(k>=60)
    delta = exp(log_gh(k)/(k-1))
    return float(delta)


def get_m(q,b):
    return 1/2. + ln(q)*(b-1)/(4*ln(gh(b)))

small_slope_t8 = {2:0.04473,3:0.04472,4:0.04402,5:0.04407,6:0.04334,7:0.04326,8:0.04218,9:0.04237,10:0.04144,11:0.04054,12:0.03961,13:0.03862,14:0.03745,15:0.03673,16:0.03585,17:0.03477,18:0.03378,19:0.03298,20:0.03222,21:0.03155,22:0.03088,23:0.03029,24:0.02999,25:0.02954,26:0.02922,27:0.02891,28:0.02878,29:0.02850,30:0.02827,31:0.02801,32:0.02786,33:0.02761,34:0.02768,35:0.02744,36:0.02728,37:0.02713,38:0.02689,39:0.02678,40:0.02671,41:0.02647,42:0.02634,43:0.02614,44:0.02595,45:0.02583,46:0.02559,47:0.02534,48:0.02514,49:0.02506,50:0.02493,51:0.02475,52:0.02454,53:0.02441,54:0.02427,55:0.02407,56:0.02393,57:0.02371,58:0.02366,59:0.02341,60:0.02332}
@CachedFunction
def slope(beta):
    if beta<=60:
        return small_slope_t8[beta]
    if beta<=70:
        # interpolate between experimental and asymptotics
        ratio = (70-beta)/10.
        return ratio*small_slope_t8[60]+(1.-ratio)*2*log(delta(70))
    else:
        return 2 * log(delta(beta))

def find_current_ncrit( profile ):
    """
    profile: squares of r_{i,i}
    """
    prof = [ log(p) for p in profile ]
    n = len(prof)
    diff = abs( prof[n-1]-prof[n-2] )
    if diff>0.5:
        return n
    for i in range(n-2,-1,-1):
        diff_new = abs( prof[i]-prof[i-1] )
        if diff_new > 10**-4 + diff:
            return i
    if i==0:
        return n

class bkzsim:

    def __init__( self, **params ):
        default_param = { 'b0': 50 ,'b1': 80 ,'step': 2 ,'t0':1 ,'t1':8 , 'estimators': "cn11 bsw18 kkn" }
        for key in default_param:
            if key in params.keys():
                default_param[key] = params[key]
        default_param[key] = default_param['estimators'].split()
        self.params = default_param
        self.ncrits = []

    def __call__( self, r, **params ):
        default_param = self.params
        for key in default_param:
            if key in params.keys():
                default_param[key] = params[key]

        b0     = default_param['b0'] #preprocessing beta
        b1     = default_param['b1']    #pump'n'jump beta
        step   = default_param['step']    #step in pump'n'jump bkz. betas = range(b0,b1,step)
        t0     = default_param['t0']    #tours in preprocessing
        t1     = default_param['t1']    #tours in pump'n'jump bkz
        estimators = default_param['estimators'].split() if isinstance( default_param['estimators'], str ) else default_param['estimators']

        output = {}

        for e in estimators:
            if e=="bsw18":
                output[e] = self.bsw( copy(r) )
            if e=="cn11":
                output[e] = self.cn11( copy(r) )
            if e=="kkn":
                output[e] = self.kknsimulator( copy(r) )
        return output
#         print(b0,b1,step,t0,t1,estimators)

    def __str__(self):
        b0     = self.params['b0'] #preprocessing beta
        b1     = self.params['b1']    #pump'n'jump beta
        step   = self.params['step']    #step in pump'n'jump bkz. betas = range(b0,b1,step)
        t0     = self.params['t0']    #tours in preprocessing
        t1     = self.params['t1']    #tours in pump'n'jump bkz
        estimators = estimators_set(self.params['estimators'])

        s = f"BKZ sim with b0={b0}, b1={b1}, step={step}, t0={t0}, t1={t1},\nEstimators: {estimators}"
        return s

    def __repr__(self):
        return self.__str__()

    def bsw(self,r):
        default_param = self.params

        b0     = self.params['b0'] #preprocessing beta
        b1     = self.params['b1']    #pump'n'jump beta
        step   = self.params['step']    #step in pump'n'jump bkz. betas = range(b0,b1,step)
        t0     = self.params['t0']    #tours in preprocessing
        t1     = self.params['t1']    #tours in pump'n'jump bkz

        flags = 0  #| BKZ.VERBOSE
        for beta in range(3 ,b0+1 ,1 ):
            print(f"bsw18: beta={beta}",end=", ")
            r = simulate_prob(r, BKZ.Param(block_size=beta, max_loops=10, flags=flags), prng_seed=144)[0]
            #r = averaged_simulate_prob(r,  BKZ.Param(block_size=beta, max_loops=t0, flags=flags), tries=10  )
            # r = r[0]

            #rcn = simulate( rcn, BKZ.Param(block_size=beta, max_loops=tnum0, flags=flags) )[0]
        print()

        for beta in range(b0,b1+1 ,step):
            r = simulate_prob(r, BKZ.Param(block_size=beta, max_loops=10, flags=flags), prng_seed=144)[0]
            # r = averaged_simulate_prob(r,  BKZ.Param(block_size=beta, max_loops=t1, flags=flags), tries=10  )
            # r = r[0]
            print(f"bsw18: beta={beta}",end=", ")

        print()
        return r

    def cn11(self,r):
        b0     = self.params['b0'] #preprocessing beta
        b1     = self.params['b1']    #pump'n'jump beta
        step   = self.params['step']    #step in pump'n'jump bkz. betas = range(b0,b1,step)
        t0     = self.params['t0']    #tours in preprocessing
        t1     = self.params['t1']    #tours in pump'n'jump bkz

        #r = [ sqrt(rr) for rr in r ]

        flags = 0  #| BKZ.VERBOSE
        for beta in range(3 ,b0+1 ,1 ):
            r = simulate( r, BKZ.Param(block_size=beta, max_loops=t0, flags=flags) )[0 ]
            print(f"cn11: beta={beta}",end=", ")
        print()

        for beta in range(b0,b1+1 ,step):
            r = simulate( r, BKZ.Param(block_size=beta, max_loops=t1, flags=flags) )[0 ]
            print(f"cn11: beta={beta}",end=", ")

        print()
        return r

    def kknsimulator(self,r):
        default_param = self.params

        b0     = self.params['b0'] #preprocessing beta
        b1     = self.params['b1']    #pump'n'jump beta
        step   = self.params['step']    #step in pump'n'jump bkz. betas = range(b0,b1,step)
        t0     = self.params['t0']    #tours in preprocessing
        t1     = self.params['t1']    #tours in pump'n'jump bkz
        n = len(r)//2
        min_blocksize = 32
        assert b0>=min_blocksize , f"b0 is too small: {b0}<{min_blocksize}"

        flags = 0  #| BKZ.VERBOSE
        q = r[0]**0.5
        slope_ = slope(min_blocksize)
        m =  round( log(q)/(2*slope_) )  #round( get_m(q,b0) )
        #left, right =  m, min(2*n, m+n-1)
        left = 0
        mid = find_current_ncrit( r )
        left = max(0, mid-m)
        right = min( 2*n, mid+m )
        # right = min( 2*n, left+2*m-1 )

        print(f"left:{left} right:{right} m={m}")
        t = r[left:right]
        tnum0, tnum1 = 1 , 8
        for beta in range(8 ,min_blocksize+1 ,8 ):
            # print(beta)
            t = kknsim.averaged_simulate_prob_(t,  BKZ.Param(block_size=beta, max_loops=tnum0, flags=flags), tries=5  )
            t = t[0]

        r[left:right] = t
        nmax=right
        for beta in range(b0,b1+1,step):
            for trs in range(tnum1):
                nmax_new = max( nmax,kknsim.find_ncrit( r,beta ) )
                if nmax_new > nmax:
                    nmax+=1
                self.ncrits.append( (beta,nmax) )
                # t = kknsim.averaged_simulate_prob_(r[:nmax],  BKZ.Param(block_size=beta, max_loops=tnum1, flags=flags), tries=5  )
                t = kknsim.simulate_prob_(r[:nmax],  BKZ.Param(block_size=beta, max_loops=1, flags=flags)  )
                t = t[0]
                r[:nmax] = t
            print(f"kkn: beta={beta}",end=", ")

        print()
        return r
