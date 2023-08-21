from fpylll import *
from keygen import NTRUKeyGenerator
from pre_processing import projectAgainstOne
from pre_processing import projection, projectAgainstOne

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
    onev = vector( [1]*n+[0]*n )
    n_ = n-1
    Bq = []
    for i in range(n_):
        qv = vector(i*[0]+[q]+(n+n-i-1)*[0])
        qv = qv - onev*qv/(n) * onev
        Bq.append( qv )

    Bq = matrix(Bq)
    denom = Bq.denominator()

    print( f"denom: {denom}" )

    Bq = denom*Bq
    Bq = Bq.change_ring(ZZ)
    B = IntegerMatrix.from_matrix( Bq )
    G = GSO.Mat( B, float_type='dd' )
    G.update_gso()

    ldenom = log(denom)  #we assume, the tail has the same shape as qary vectors
    lgq = log(q)
    l = [ G.get_r(i,i)  for i in range(n_) ] + [ G.get_r(i,i)/q^2 for i in range(n_) ]

    return l, denom

class bkzsim:

    def __init__( self, **params ):
        default_param = { 'b0': 50,'b1': 80,'step': 2,'t0':1,'t1':8, 'estimators': "cn11 bsw18 kkn" }
        for key in default_param:
            if key in params.keys():
                default_param[key] = params[key]
        default_param[key] = default_param['estimators'].split()
        self.params = default_param


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

        flags = 0 #| BKZ.VERBOSE
        for beta in range(3,b0+1,1):
            print(f"bsw18: beta={beta}",end=", ")
            #r = simulate_prob(r, BKZ.Param(block_size=beta, max_loops=10, flags=flags), prng_seed=144)[0]
            r = averaged_simulate_prob(r,  BKZ.Param(block_size=beta, max_loops=t0, flags=flags), tries=10 )
            r = r[0]

            #rcn = simulate( rcn, BKZ.Param(block_size=beta, max_loops=tnum0, flags=flags) )[0]
        print()

        for beta in range(b0,b1+1,step):
            #r = simulate_prob(r, BKZ.Param(block_size=beta, max_loops=10, flags=flags), prng_seed=144)[0]
            r = averaged_simulate_prob(r,  BKZ.Param(block_size=beta, max_loops=t1, flags=flags), tries=10 )
            r = r[0]
            print(f"bsw18: beta={beta}",end=", ")

            #rcn = simulate( rcn, BKZ.Param(block_size=beta, max_loops=tnum0, flags=flags) )[0]

        print()
        return r

    def cn11(self,r):
        b0     = self.params['b0'] #preprocessing beta
        b1     = self.params['b1']    #pump'n'jump beta
        step   = self.params['step']    #step in pump'n'jump bkz. betas = range(b0,b1,step)
        t0     = self.params['t0']    #tours in preprocessing
        t1     = self.params['t1']    #tours in pump'n'jump bkz

        #r = [ sqrt(rr) for rr in r ]

        flags = 0 #| BKZ.VERBOSE
        for beta in range(3,b0+1,1):
            r = simulate( r, BKZ.Param(block_size=beta, max_loops=t0, flags=flags) )[0]
            print(f"cn11: beta={beta}",end=", ")
        print()

        for beta in range(b0,b1+1,step):
            r = simulate( r, BKZ.Param(block_size=beta, max_loops=t1, flags=flags) )[0]
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
        assert b0>=45, f"b0 is too small: {b0}<45"

        flags = 0 #| BKZ.VERBOSE
        left, right = max(50,n-2*b0), min(2*n-50,n+2*b0)
        t = r[left:right]
        tnum0, tnum1 = 1, 8
        for beta in range(5,46,5):
            print(beta)
            t = kknsim.averaged_simulate_prob_(t,  BKZ.Param(block_size=beta, max_loops=tnum0, flags=flags), tries=10 )
            t = t[0]

        r[left:right] = t

        print()

        nmax = min( 2*n-50, n+b0 )
        for beta in range(b0,b1+1,step):
            #r = simulate_prob(r, BKZ.Param(block_size=beta, max_loops=10, flags=flags), prng_seed=144)[0]
            nmax = max( nmax,kknsim.find_ncrit( r,beta ) )
            t = kknsim.averaged_simulate_prob_(r[:nmax],  BKZ.Param(block_size=beta, max_loops=tnum1, flags=flags), tries=10 )
            t = t[0]
            r[:nmax] = t
            print(f"kkn: beta={beta}",end=", ")

            #rcn = simulate( rcn, BKZ.Param(block_size=beta, max_loops=tnum0, flags=flags) )[0]

        print()
        return r