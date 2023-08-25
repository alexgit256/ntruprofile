from keygen import NTRUKeyGenerator
import pre_processing
from pre_processing import projectAgainstOne
import os, re, time
from pathlib import Path
from fpylll import *
from time import perf_counter
from sage.stats.distributions.discrete_gaussian_integer import DiscreteGaussianDistributionIntegerSampler

from fpylll import *
from fpylll import BKZ as BKZ_FPYLLL
from fpylll import LLL as LLL_FPYLLL
from fpylll import GSO, IntegerMatrix, FPLLL, Enumeration, EnumerationError, EvaluatorStrategy
from fpylll.tools.quality import basis_quality
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.util import gaussian_heuristic
from itertools import chain
import time
from time import perf_counter
from copy import deepcopy

import numpy as np

import contextlib
try:
    from multiprocess import Pool  # you might need pip install multiprocess
except ModuleNotFoundError:
    from multiprocessing import Pool

FPLLL.set_precision(200)
RR = RealField(200)

def enum_block( basis,beta,shift=0 ):
    filename = "exp_shift_"+basis
    print(f"Currently in {shift}-{basis}")
    round_time = perf_counter()
    try:
        with open(filename, 'w') as file:
            with contextlib.redirect_stdout(file):
                rootdir = "./basis_dumps"
                M = IntegerMatrix.from_file(rootdir+"/"+basis)
                n = M.nrows//2
                left, right = 2*n-beta-shift, 2*n-shift
                print(n,beta,left,right)

                G = GSO.Mat( M, float_type="mpfr" )
                G.update_gso()

                r = [ log(G.get_r(i,i))/2. for i in range(2*n) ]

#                 list_plot(r).show(xmin=left)

                Mu = matrix([
                   [ ( G.get_r(i,j)/G.get_r(j,j)^0.5 ) if j<=i else 0 for j in range(left,right) ] for i in range(left,right)
                ])

                s = 2**32
                B = matrix(ZZ,[
                    [ round(s*Mu[i,j]) for j in range(Mu.ncols()) ] for i in range(Mu.nrows())
                ])

                G = GSO.Mat( IntegerMatrix.from_matrix(B), float_type="ld" )
                G.update_gso()

                l = [ log(G.get_r(i,i)) for i in range(B.nrows()) ]

                t0,t1 = gaussian_heuristic([G.get_r(i,i) for i in range(B.nrows())])^0.5 , norm(B[0]).n() #min([norm(bb) for bb in B])
                print( t0, t1, t1/t0 )

#                 list_plot(l).show()

                lll_red = LLL_FPYLLL.Reduction(G,delta=0.98, eta=0.51)
                print(f"Launching LLL... initial log r00")
                then = time.perf_counter()
                lll_red()
                print(f"lll done in {time.perf_counter()-then}  log r00: {RR(log( lll_red.M.get_r(0,0),2 )/2)}")

                B = matrix(B)
                B = matrix(list(filter(lambda bb:norm(bb)>0, B)))

                l = [log( G.get_r(i,i) )/2 for i in range(B.nrows())]

#                 list_plot(l).show()
                GSO_M=GSO.Mat(IntegerMatrix.from_matrix(B),float_type="mpfr")
                GSO_M.update_gso()
                bkz = BKZReduction(GSO_M)

                flags = BKZ_FPYLLL.AUTO_ABORT|BKZ_FPYLLL.MAX_LOOPS|BKZ_FPYLLL.GH_BND

                print( min([norm(bb).n() for bb in B]) )
                for beta in range(4,53,1):    #BKZ reduce the basis
                    par = BKZ_FPYLLL.Param(beta,
                                           max_loops=14,
                                           flags=flags
                                           )
                    then_round=time.perf_counter()
                    bkz(par)
                    round_time = time.perf_counter()-then_round
                    print( f"BKZ-{beta} done in {round_time}" )

                l = [log( G.get_r(i,i) )/2 for i in range(B.nrows())]
                print( min([norm(bb).n() for bb in B]) )

#                 list_plot(l).show()

                then = perf_counter()
                enum = Enumeration(GSO_M, strategy=EvaluatorStrategy.BEST_N_SOLUTIONS, sub_solutions=True, nr_solutions=2)
                _ = enum.enumerate(0, B.nrows(), 1.05*GSO_M.get_r(0, 0), 0)
                print(f"Enum done in {perf_counter()-then}")
                print(_)
                print(RR(_[0][0]^0.5))
    except Exception as err:
        print(f"Error:{err} at shift {shift}")
        return None
    print(f"Done {shift}-{basis}")  #in {perf_counter()-round_time, round_time-perf_counter()}
    return t0, t1, RR(sqrt( _[0][0] ))

def run_all( n,beta,max_shift,nthreads=20 ):
    bases = []
    filename = f"n_{n}_lattype_classic_b_{beta}_"

    #rootdir = os.curdir+"/g6kdump" #get drname stupid way
    rootdir = "./basis_dumps"
    regex = re.compile(filename+"*")
    print(filename+"*")

    data_folder = Path("./basis_dumps")

    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if regex.match(file):
                print(file)
                bases.append( file )
    output = []
    pool = Pool(processes = nthreads )
    tasks = []

    for shift in range(max_shift):
        for basis in bases:
            tasks.append( pool.apply_async(
            enum_block, (basis,beta,shift)
            ) )
    for t in tasks:
        output.append( t.get() )

    pool.close()
    return output

out = run_all( n=231,beta=70,max_shift=20,nthreads=16 )   #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

info = {"||b0||/gh":[],
        "||s||/gh":[],
        "||b0||/||s||":[]
       }

svpevents = 0
for o in out:
    info["||b0||/gh"].append( o[1]/o[0] )
    info["||s||/gh"].append( o[2]/o[0] )
    info["||b0||/||s||"].append( o[2]/o[1] )
    if o[2]/o[1] < 1:
        svpevents +=1

print(info)
print()

print(f"Non reduced bases: {svpevents} out of {len(out)}")
print(f"That is {svpevents/len(out)*100.}%")

info["||b0||/gh"] = np.mean(info["||b0||/gh"])
info["||s||/gh"] = np.mean(info["||s||/gh"])
info["||b0||/||s||"] = np.mean(info["||b0||/||s||"])
print( info )
