"""Module greedy_agent -- defines the strategy of a greedy agent in the retailer's environment.

Class GreedyAgent:
    Defines how a greedy agent tackles procurement, and implements a simulation based on environment.
    
"""

import numpy as np
from pyscipopt import Model


class GreedyAgent: 

    def __init__(self, environment, basestock = 'auto'): 
        self.env = environment
        if not isinstance(basestock, (int, type(None))): 
            assert basestock == 'auto', 'order_quantity must be integer, "auto" or None' # None = total quantity per period
            self.basestock = self.env.basestock_level()
        else: 
            self.basestock = basestock #integer or None
        self.model = None 
        self.prices = self.env.get_available_prices()
        self.index_counter_best, self.index_counter_only = 0, 0 #initiate in case self.action is called explicitly 
        self.budget_counter_best, self.budget_counter_only = 0, 0
        self.last_resort_counter = 0 
        
    def test(self, epochs = 1): 
        total_costs, average_price_paid, average_demand_price = [], [], []
        index_counter_best_list, index_counter_only_list = [], [] 
        budget_counter_best_list, budget_counter_only_list = [], [] 
        for e in range(epochs):
            print("epoch starts:", e)
            self.index_counter_best, self.index_counter_only = 0, 0 #counter per method call 
            self.budget_counter_best, self.budget_counter_only = 0, 0
            self.last_resort_counter = 0 
            epoch_costs = 0
            last_state, isFinished = self.env.env_start()
            while not isFinished:
                last_action = self.action(last_state)
                reward, state, isFinished = self.env.env_step(last_action)
                print('budget:', state[1])
                epoch_costs -= reward
                last_state = state
            total_demand = sum(self.env.get_all_demands())
            total_costs.append(round(epoch_costs, 2))
            average_price_paid.append(last_state[-2]/ last_state[-1])
            average_demand_price.append(np.dot(total_demand,self.prices)/np.sum(total_demand))
            index_counter_best_list.append(self.index_counter_best)
            index_counter_only_list.append(self.index_counter_only)
            budget_counter_best_list.append(self.budget_counter_best)
            budget_counter_only_list.append(self.budget_counter_only)
        print('the basestock level is', self.basestock)
        print('total costs:', total_costs, 'average price paid:', average_price_paid, 'average demanded price', average_demand_price)
        return {'average_price_paid': average_price_paid, 'average_demand_price': average_demand_price, 'constraints': {'index_contraint_best': index_counter_best_list, 'index_contraint_only': index_counter_only_list, 'budget_contraint_best':  budget_counter_best_list, 'budget_constraint_only': budget_counter_only_list}}        
    
    def action(self, state): 
        """ returns array of quantites (shape len(av_prices),), according to greedy policy """
        total_quantity = self.total_order_quantity(state)
        best_quantities = self.greedy_quantities(total_quantity, state)
        return best_quantities
      

    def total_order_quantity(self, state): 
        """ follows a basestock policy, returns total quantity to order in current period """
        curr_inventory_level = sum(state[0])
        return max(self.basestock- curr_inventory_level, 0)
        
        
    def greedy_quantities(self, tot_quantity, state): 
        """ returns np.array of best quantites according to greedy rule"""
        if tot_quantity == 0: 
            return np.zeros(len(self.prices))
        candidate_quantities =  self.index_quantities(tot_quantity, state[4], state[5])
        if candidate_quantities is not None: 
            new_budget = self.env.candidate_budget(candidate_quantities) #existing budget + updated budget with candidate quantities 
            if new_budget <= 0:
                self.index_counter_best += 1
                return candidate_quantities
        demand_list = self.env.demand
        better_quantities = self.budget_quantities(tot_quantity, demand_list, state[1])
        if better_quantities is not None: 
            if candidate_quantities is None: 
                self.budget_counter_only += 1
            else: 
                self.budget_counter_best += 1
            return better_quantities
        if candidate_quantities is not None: 
            self.index_counter_only += 1
            return candidate_quantities 
        self.last_resort_counter += 1 
        return self.last_resort_quantities(tot_quantity, demand_list) 
        
    def budget_quantities(self, tot_quantity, demands, budget): 
        model=Model('budget_solver')
        model.hideOutput()
        model, var = self._integer_optim_setup(model)
        purch_costs=0
        new_budget = 0
        for price, quant, demand in zip(self.prices, var, demands): 
            purch_costs += price*quant 
            new_budget -= price*(demand - quant) 
        model.setObjective(purch_costs, "minimize")
        model.addCons(sum(var)==tot_quantity)
        model.addCons(new_budget >= -float(budget))
        model.optimize()
        if model.getStatus() == 'optimal':
            sol = model.getBestSol()
            return np.array([round(sol[var[i]],2) for i in range(len(self.prices))])   
        
    def index_quantities(self, tot_quantity, purch_val, purch_vol): 
        """ returns np.array of best quantites according to index contraint"""
        model = Model('index_solver')
        model.hideOutput()
        model, var = self._integer_optim_setup(model)
        I = self.env.index
        purch_costs=0
        for price, quant in zip(self.prices, var): 
            purch_costs += price*quant 
        model.setObjective(purch_costs, "minimize")
        model.addCons(purch_val + purch_costs >= I*(purch_vol + tot_quantity))
        model.addCons(sum(var)==tot_quantity)
        model.optimize()
        if model.getStatus() == 'optimal':
            sol = model.getBestSol()
            return np.array([round(sol[var[i]],2) for i in range(len(self.prices))])

    def last_resort_quantities(self, tot_quantity, demands): 
        model = Model('last_resort')
        model.hideOutput()      
        model, var = self._integer_optim_setup(model)
        new_budget = 0
        for price, quant, demand in zip(self.prices, var, demands): 
            new_budget -= price*(demand - quant)         
        model.setObjective(new_budget, "maximize")
        model.addCons(sum(var)==tot_quantity)
        model.optimize()
        sol = model.getBestSol()
        return np.array([round(sol[var[i]],2) for i in range(len(self.prices))])

    def _integer_optim_setup(self, model): 
        quantities_names = ['q{}'.format(i) for i in range(len(self.prices))]
        var = []
        for name in quantities_names: 
            q = model.addVar(name, vtype='INTEGER')
            var.append(q)
        return model, var






