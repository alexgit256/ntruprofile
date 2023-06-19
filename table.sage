"""
To run simulator on CS lattice with n=211 and beta=60..100 type:
    run( n=211, q=4096, bpre=60, bpost=100,  projected=False )
For projective lattice type:
    run( n=221, q=4096, bpre=50, bpost=92,  projected=True )
Currently avaliable data:
    n=211, bpost = 60:100:2, projective=False
    n=221, bpost = 60:100:2, projective=False
    n=221, bpost = 60:92:2, projective=True
"""
import os
import re
import numpy as np

from fpylll.tools.bkz_simulator import  simulate, simulate_prob, averaged_simulate_prob
from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ
from copy import deepcopy

def checkDSD(r):
    """
        r: set of log(|r_i_i|) (or log( |r_i_i|^2 ), it does not affect the correctness)
        checks if the DSD event happens
    """
    s_1 = 0
    s_2 = 0
    n = len(r)
    for i in range(n):
      norm = r[i]
      if i < n/2:
        s_1 += norm
      else:
        s_2 += norm

    return s_1 <= s_2

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
    G = GSO.Mat( B, float_type='dd' )
    G.update_gso()

    ldenom = log(denom)  #we assume, the tail has the same shape as qary vectors
    lgq = log(q)
    l = [ G.get_r(i,i)  for i in range(n_) ] + [ G.get_r(i,i)/q**2  for i in range(n_) ]

    return l, denom

def run(n=211, bpre=50, bpost=70, q=4096, projected=False):
    RR = RealField(100)

    scale_factor = 1
    lscale_factor = 0
    if projected is True:
        projected_profile,scale_factor  = phione_profile(n,q)
        lscale_factor = log(scale_factor).n()

    profiles = []
    lattype = "phi_projected" if projected else "classic"
    filename = f"n_{n}_lattype_{lattype}_b_{bpost}_seed"
    # filename=""

    #rootdir = os.curdir+"/g6kdump" #get dirname stupid way
    rootdir = "./g6kdump"
    regex = re.compile(filename+"*")

    from pathlib import Path

    data_folder = Path("./g6kdump")

    for root, dirs, files in os.walk(rootdir):
        for file in files:
            if regex.match(file):
                print(file)
                file_to_open = data_folder / file
                with open(file_to_open,"r") as file:
                    profiles.append( file.read() )
    profiles_ = []
    for i in range(len(profiles)):
        profiles[i] = eval( profiles[i] )
        profiles[i] = [ log(p)/2 for p in profiles[i]  ]
        if sum( profiles[i][:n] ) < sum( profiles[i][n:] ) or len(profiles[i])<(2*n if not projected else 2*n-5):
            print(f"DSD event at i={i}")
            continue
        profiles_.append(profiles[i])

    if len(profiles)==0:
        print("No practical data found. Aborting...")
        return

    profiles = profiles_
    if len(profiles)==0:
        print("No practical data found. Aborting...")
        return

    l = list( np.average( profiles, axis=0 ) )  #mean profile

    l = [ ll - lscale_factor for ll in l ]

    tnum0, tnum1 = 5, 8  #pre-bkz and pump_n_kump bkz tours

    if not projected:
        r = n*[ float(q)^2 ] + n*[ 1 ]
    else:
        r = [ pp / scale_factor^2 for pp in projected_profile ]
    rcn = deepcopy ( r )

    flags = 0 #| BKZ.VERBOSE
    for beta in range(7,bpre+1,1):
        #print(beta)
        r = averaged_simulate_prob(r,  BKZ.Param(block_size=beta, max_loops=tnum0, flags=flags), tries=10 )
        r = r[0]

        rcn = simulate( rcn, BKZ.Param(block_size=beta, max_loops=tnum0, flags=flags) )[0]


    for beta in range(50,bpost+1,2):
        r = averaged_simulate_prob(r,  BKZ.Param(block_size=beta, max_loops=tnum1, flags=flags), tries=10 )
        r = r[0]

        rcn = simulate( rcn, BKZ.Param(block_size=beta, max_loops=tnum1, flags=flags) )[0]
        print(f"simulating beta={beta}...",end=", ")
    print()

    r = [ log(rr)/2 for rr in r ]
    rcn = [ log(rr)/2 for rr in rcn ]

    if projected:
        n = len(l)//2

    rounding = RealField(28)
    pract, bswest, cnest =  rounding(sum( l[n:] )) , rounding(sum(r[n:])), rounding(sum(rcn[n:]))
    p0 =  list_plot(l, figsize=12, legend_label='$Practice$')
    p1 =  list_plot(r, color="red", figsize=12, legend_label='$BSW18$', pointsize=15,) #BSW18
    p2 = list_plot(rcn,color="green",legend_label='$CN11$', title=f'n={n}, beta={beta}, lattype={lattype}\n prac={pract}, bswest={bswest}, cnest={cnest}')  #CN11

    P = (p0+p1+p2)
    P.save_image( f"ntru_n{n}_b{bpost}_{lattype}.png" )

#     for i in range(2*n):
#         print( f"{l[i]} {r[i]} {rcn[i]}" )

    print("Avg experiment:")
    print(sum(l[n:]))
    print("BSW18 estimate:")
    print(sum(r[n:]))
    print("CN11 estimate:")
    print(sum(rcn[n:]))
    P.show()