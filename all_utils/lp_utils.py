'''
Created on Oct 16, 2020

@author: Atrisha
'''


import numpy as np
import matplotlib.pyplot as plt
from pulp import LpMaximize, LpProblem, LpVariable
import pulp as pl
from pulp.constants import LpMinimize
from pulp import *

def demo():

    x_1 = np.linspace(0, 30, 1000)
    x_2 = np.linspace(0, 30, 1000)
    
    # plot
    fig, ax = plt.subplots()
    fig.set_size_inches(14.7, 8.27)
    
    # draw constraints
    plt.axvline(7, color='g', label=r'$x_1 \geq 7$') # constraint 1
    plt.axhline(8, color='r', label=r'$x_2 \geq 8$') # constraint 2
    plt.plot(x_1, (2*(x_1)), label=r'$x_2 \leq 2x_1$') # constraint 3
    plt.plot(x_1, 25 - (1.5*x_1), label=r'$1.5x_1 + x_2 \leq 25$') # constraint 4
    
    
    plt.xlim((0, 25))
    plt.ylim((0, 30))
    plt.xlabel(r'Number of keyboards ($x_1$)')
    plt.ylabel(r'Number of mice ($x_2$)')
    
    # fill in the fesaible region
    plt.fill_between(x_1, np.minimum(25 - (1.5*x_1), (2*(x_1))), np.minimum(25 - (1.5*x_1), 8), 
                     where=x_1 >= 7,
                     color='green', alpha=0.25)
    plt.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.show()
    
    
    
    # first create a model object
    model = LpProblem(name="computer_parts_problem", sense=pl.LpMinimize)
    
    # declare our variables (note we are setting the low bound here so we dont have to add them as constraints)
    # note we set the variable type to int so we dont get flots in solution (not practical for production)
    x_1 = LpVariable(name="x_1", lowBound=7, cat="Integer")
    x_2 = LpVariable(name="x_2", lowBound=8, cat="Integer")
    
    # set the objective function for the model
    model += (50 * x_1) + (37 * x_2)
    
    # now we can add our constraints
    model += (x_2 <= 2 * x_1, "supply_and_demand")
    model += ((1.5 * x_1) + x_2 <= 25, "labour_hours")
    
    # take a look at the model
    print(model)
    
    solution = model.solve()
    
    # print the solution
    for variable in model.variables():
        print(f"Optimal value for {variable.name} is {variable.value()}")
        
    print(f"\nThis will yield a total profit of {model.objective.value()}")
    

def solve_lp_multivar(varstr_list,obj_coeff,constr_list):
    var_list = []
    for v in varstr_list:
        w = LpVariable(v,0,1)
        var_list.append(w)
        exec(v+'=w')
    prob = LpProblem("max_bounds", LpMaximize)
    opt_expr = ''
    for idx,lpv in enumerate(varstr_list):
        opt_expr += str(varstr_list[idx])+'*'+str(sum(obj_coeff[idx]))+'+'
    opt_expr = eval(opt_expr[:-1])
    prob += opt_expr
    f=1
    
    for constr in constr_list:
        constr_expr = ''
        constr_expr = '+'.join([str(v)+'*'+str(c) for v,c in zip(constr[0],constr[1])])
        constr_expr += '>= 0'
        constr_expr = eval(constr_expr)
        prob += constr_expr
    eq_constr = '+'.join(varstr_list) + '== 1'
    eq_constr = eval(eq_constr)
    prob += eq_constr
    pl.LpSolverDefault.msg = False
    status = prob.solve()
    if status != -1:
        solns = [x.varValue for x in var_list]
    else:
        solns = None
    return solns
            
    

def solve_lp(obj,constr,num_params):
    
    w_1 = LpVariable("w1", 0, 1)
    w_2 = LpVariable("w2", 0, 1)
    if num_params == 3:
        w_3 = LpVariable("w3", 0, 1)
    
    prob = LpProblem("max_bounds", LpMaximize)
    if num_params == 3:
        prob += w_1*obj[0] + w_2*obj[1] + w_3*obj[2]
    else:
        prob += w_1*obj[0] + w_2*obj[1]
    
    for c in constr:
        if num_params == 3:
            prob += w_1*c[0] + w_2*c[1] + w_3*c[2] >= 0
        else:
            prob += w_1*c[0] + w_2*c[1] >= 0
        
    if num_params == 3:
        prob += w_1 + w_2 + w_3 == 1
    else:
        prob += w_1 + w_2== 1
    #print(prob)
    pl.LpSolverDefault.msg = False
    try:
        status = prob.solve()
    except:
        status = -1
    if status != -1:
        
        #for variable in prob.variables():
        #    print(f"Optimal value for {variable.name} is {variable.value()}")
        if num_params == 3:
            upp_bounds = [w_1.varValue,w_2.varValue,w_3.varValue,0]
        else:
            upp_bounds = [w_1.varValue,w_2.varValue,0,0]
    else:
        upp_bounds = [None]*4
    prob = LpProblem("max_bounds", LpMinimize)
    if num_params == 3:
        prob += w_1*obj[0] + w_2*obj[1] + w_3*obj[2]
    else:
        prob += w_1*obj[0] + w_2*obj[1]
    
    for c in constr:
        if num_params == 3:
            prob += w_1*c[0] + w_2*c[1] + w_3*c[2] >= 0
        else:
            prob += w_1*c[0] + w_2*c[1] >= 0
        
    if num_params == 3:
        prob += w_1 + w_2 + w_3 == 1
    else:
        prob += w_1 + w_2== 1
    #print(prob)
    pl.LpSolverDefault.msg = False
    try:
        status = prob.solve()
    except:
        status = -1
    if status != -1:
        if num_params == 3:
            low_bounds = [w_1.varValue,w_2.varValue,w_3.varValue,0]
        else:
            low_bounds = [w_1.varValue,w_2.varValue,0,0]
    else:
        low_bounds = [None]*4
    return list(zip(upp_bounds,low_bounds))