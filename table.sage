import os
import re
import numpy as np

from fpylll.tools.bkz_simulator import  simulate, simulate_prob, averaged_simulate_prob
from fpylll import IntegerMatrix, GSO, LLL, FPLLL, BKZ
from copy import deepcopy

from bkz_sim import bkzsim
FPLLL. set_precision(200)

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
    param n: half the dimension of NTRU lattice
    param q: modulus
    """
    RR = RealField(144)

    onev = vector( [1 ]*n+[0 ]*n )
    n_ = n-1
    Bq = []
    for i in range(n_): #we project the qary lattice
        qv = vector(i*[0 ]+[q]+(n+n-i-1 )*[0 ])
        qv = qv - onev*qv/(n) * onev
        Bq.append( qv )
    Bq = matrix(Bq)
    denom = Bq.denominator()

    print( f"denom: {denom}" )

    Bq = denom*Bq   #we scale the lattice as in ntru_with_sieving
    Bq = Bq.change_ring(ZZ)
    B = IntegerMatrix.from_matrix( Bq )
    G = GSO.Mat( B, float_type='mpfr' )
    G.update_gso()

    # first_vector_norm = sqrt((q*(n-2))^2+(n-2)*q^2)/(n-1) * (n-1) #scaled

    qary = [ G.get_r(i,i)  for i in range(n-3) ]    #we get the qary profile
    D = ( log(q)*(n-3)-log(n-1) + (2*n-4)*log(n-1) ).n() # the log determinant we expect to see
    ones = [ RR(G.get_r(i,i))/q^2  for i in range(n-1) ]
    # We scale the qary profile for it to resemble the profile of the second "half" of the vectors. This is
    # somewhat correct since qary vectors projected against (1,0) span the same projective lattice B[0:n-2]
    # and the last n-2 GS vectors before projecting b_{n-2},...,b_{2*n-5} against (0,1) are of the form (0,...,1,...,0).
    # We don't chase the authentic initial profile since it does not exist: the vectors of proj_{(0,1)}( proj_{(1,0)}( B_\Phi ) )
    # are linearlly dependent which cannot be resolved without distorting the r_{i,i} for all i < 2*n-4.

    l = qary + ones
    logl = [log(ll) for ll in l]
    D2 = sum(logl)
    D_over_D2_distributed = exp( (2*D-D2)/len(l) )

    l = [float(ll*D_over_D2_distributed) for ll in l]
    #print(f"Vol predicted:{D} vs generated:{sum([log(ll) for ll in l])/2}")

    return l, denom

def run(n=211, bpre=50, bpost=70, step=2, q=None, projected=False):
    RR = RealField(100)

    if q is None:
        q = 2**ceil(3.5 + log(n,2)) #HRSS
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
    #rcn = deepcopy ( r )
    #rkkn = deepcopy( r )

    bkz = bkzsim( b0=bpre, b1=bpost, step=step, t0=5, t1=8, estimators="cn11 bsw18 kkn" )
    out = bkz(r)
    rbsw, rcn, rkkn = out["bsw18"], out["cn11"], out["kkn"]
    """
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
    """
    rbsw = [ log(rr)/2 for rr in rbsw ]
    rcn  = [ log(rr)/2 for rr in rcn ]
    rkkn = [ log(rr)/2 for rr in rkkn ]

    if projected:
        n = len(l)//2

    rounding = RealField(28)
    pract, bswest, cnest, kknest =  rounding(sum( l[n:] )) , rounding(sum(rbsw[n:])), rounding(sum(rcn[n:])), rounding(sum(rkkn[n:]))
    p0 =  list_plot(l, figsize=12, legend_label='$Practice$')
    p1 =  list_plot(rbsw, color="red", figsize=12, legend_label='$BSW18$', pointsize=15,) #BSW18
    p2 = list_plot(rcn,color="green",legend_label='$CN11$', title=f'n={n}, beta={bpost}, lattype={lattype}\n prac={pract}, bswest={bswest},\n cnest={cnest}, kknest={kknest}')  #CN11
    p3 = list_plot(rkkn,color="black",legend_label='New estimation' )

    P = (p0+p1+p2+p3)
    P.save_image( f"ntru_n{n}_b{bpost}_{lattype}.png" )

#     for i in range(2*n):
#         print( f"{l[i]} {r[i]} {rcn[i]}" )

    print("Avg experiment:")
    print(sum(l[n:]))
    print("BSW18 estimate:")
    print(sum(rbsw[n:]))
    print("CN11 estimate:")
    print(sum(rcn[n:]))
    print("KKN estimate:")
    print(sum(rkkn[n:]))
    #print(f"Check. EXP:{sum(l)} BSW18:{sum(rbsw)} CN11:{sum(rcn)} KKN:{sum(rkkn)}")  #check if the profiles sum up to a same value
    P.show()

    (p0+p3).show(figsize=14)
