from equilibria.equilibria_estimation import build_analysis_table
import sys
import itertools
import constants

file_list = [sys.argv[1]]
l1_eq_list = ['NASH','BR','MAXMIN']
l3_eq_list = ['BR','MAXMIN']
traj_gen_list = ['BOUNDARY','GAUSSIAN']
all_params = list(itertools.product(file_list,traj_gen_list,l1_eq_list,l3_eq_list))
for f in file_list:
    all_params = all_params + [(f,'BASELINE','NASH',None)]
    all_params = all_params + [(f,'BASELINE','BR',None)]
    all_params = all_params + [(f,'BASELINE','MAXMIN',None)]
for parms in all_params:
    constants.TRAJECTORY_TYPE = parms[1]
    constants.L1_EQ_TYPE = parms[2]
    constants.L3_EQ_TYPE = parms[3]
    build_analysis_table(False,True,parms[0])

