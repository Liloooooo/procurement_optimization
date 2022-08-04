"""Module environment -- defines the retailer's environment.

Class ProcurementEnvironment:
    Full definition of operations environment
    
"""


import numpy as np
from datetime import datetime, timedelta, date
import calendar
import holidays
import pandas as pd 
from scipy.stats import gamma
from fitter import Fitter


class ProcurementEnvironment:
    """Retailer's environment
    
    Attrs:
        data (pandas.DataFrame): past demand data for different prices, optional (alternative: simulation) 
        key: env_info (dictionnary): defines environment parameters: 
                                'start_date' (string in the form ddmmyy) 
                                'VLT' (int): vendor lead time, 
                                'initial_stock' (float, int): initial inventory level for item, 
                                'supplier index' (float): average price that supplier allows; if true average price is lower, retailer must provide documentation, and eventually rebalance difference and pay a fine 
                                'overbudget_punishment': punsishment in case that retailer's average price is lower than index and budget is negative, 
                                'service_level': share of customers that retailer wants to serve immediately, 
        self.start_date (string)
        self.overbudget_punishment (float)
        self.service_level (float)
        self.index (float)
        self.holiday_list (list): list that comprises all holidays in years to be considered
        self.prices (np.array): from dataframe if given
        self.holding_costs (float): stockage costs 
        self.backorder_costs (float): costs for late service
        self.inventory (np.array) of length VLT + 1  
        self.current_date (date object)
        
                                

    Methods: 
        is_terminal() (bool): whether epoch is finished 
        get_all_demands(): all demands since initialization
        get_current_demand(): get demand from self.data if given 
        get_available_prices(): get all prices from data is given 
        update_budget(action): updates budget after demand has realized and action is takes (=quantities are ordered)
        update_inventory_level() 
        get_backlogging_costs()
        get_holding_costs()
        get_reward(action)
        get_orderdays_to_monthend()
        env_start(): is called when period starts, before agent takes his first action 
        env_step(action): is called every time after an action is taken 
        candidate_budget(action)
        basestock_level(): optimal basestock policy which is derived based on observed demand 
        
    """
    
    def __init__(
        self, dataset=None, env_info={}
    ):  
        if dataset is not None: 
            assert isinstance(dataset, pd.DataFrame)
            self.data = dataset
            self.data.index=pd.to_datetime(self.data.index)
            self.data.index=pd.Series(self.data.index).dt.date
        try:             
            self.start_date = datetime.strptime(env_info.get('start_date', '010119'), "%d%m%y").date()
        except ValueError:
            raise AssertionError('start_date must be of form DDMMYY.')
        assert isinstance(env_info.get('VLT', 1), int), 'VLT must be integer'
        assert isinstance(env_info.get('initial_stock', 20), (float, int)), 'initial_stock must be float or int'
        assert isinstance(env_info.get("overbudget_punishment", 0), (float, int))
        self.overbudget_punishment = env_info.get("overbudget_punishment", 0)
        assert isinstance(env_info.get("service_level", 0.6), (float, int))
        self.service_level = env_info.get("service_level", 0.6) 
        assert 0 <= self.service_level <= 1
        assert isinstance(env_info["supplier_index"], (float, int))
        self.index = env_info["supplier_index"]
        self.holiday_list = [] 
        for freeday in holidays.Germany(years=[i for i in range(2018, 2022)]).items(): 
            self.holiday_list.append(freeday[0])
        self.prices = self.get_available_prices()  
        self.holding_costs = self.prices[0] * 0.06 / 365
        self.backorder_costs = self.holding_costs * self.service_level/(1-self.service_level)
        self.inventory = np.zeros(env_info["VLT"] + 1)
        self.inventory[0] = env_info.get('initial_stock', 20)
        self._demand_track = [] 
        self.last_date = None
        
    def is_terminal(self): 
        if self.current_date.year > self.start_date.year:
            self.start_date += timedelta(days=365)
            return True
        return False
    
    def get_all_demands(self): 
        return self._demand_track

    def get_current_demand(self):
        """

        get demand array at time self.current_date, which is only on Tuesdays and Thursdays, thus, demand is accumulated up to these days  
        returns: array of shape (2,): demand per group at current time

        """
        if self.data is not None: 
            try: 
                if self.last_date is None: 
                    return np.array(self.data.loc[self.current_date, :])
                else: 
                    past_date = self.last_date + timedelta(days=1)
                    print(past_date)
                    demand = np.array(self.data.loc[past_date, :])
                    while past_date < self.current_date: #accumulated demand to Tuesdays and Thursdays
                        past_date += timedelta(days=1)
                        demand += np.array(self.data.loc[past_date, :])
                    self._demand_track.append(demand)
                    return demand
            except KeyError: 
                raise IndexError('Datetime object of current date not found in data provided. Not sufficient data?')
                
        gamma_shape = np.array((1/8, 1/8, 1/8, 1/8)) #else simulate 
        scale = np.array((20, 5, 8, 4))     
        demands = [round(np.random.gamma(gamma_shape[i], scale[i])) for i in range(4)]
        self._demand_track.append(demands)
        return np.array(demands)

    def get_available_prices(self):
        """Get available prices for self.item_no.

        Returns: arr of available prices for a product (here: 2: standard, and contractual customer)
        """
        
        if self.data is not None: 
            prices = [float(entry) for entry in self.data.columns]
            return np.array(prices)
        price_high, price_1, price_2, price_3 = 18.1, 12.8, 11.2, 14.3 #else simulate
        return np.array([price_high, price_1, price_2, price_3])

    def update_budget(self, action):
        """
        updates budget array when demand occurs by buyer 2 (contractual buyer)
        Returns: None

        """
        
        self.budget += (
            - np.dot(self.demand, self.prices) + np.dot(self.prices, action)
        )  # budget refers to ordered items (inventory in transit)

    def update_inventory_level(self):
        """
        updates inventory list after arrivals are checked for in every period, and before new orders are placed
        happens every day (datetime)
        Returns: Nothing
        """

        self.inventory[0] = self.inventory[0] + self.inventory[1] - np.sum(self.demand)
        for i in range(1, len(self.inventory) - 1):
            self.inventory[i] = self.inventory[i + 1]
        self.inventory[-1] = 0

    def get_backlogging_costs(self):
        """
        see backlooging formula ### alternative approach could be to assume decreasing demand if backlogs occur
        input:
            overdue demand: float = self.inventory[0]
            service_level=self.service_level
        returns: float


        """
        b = (
            self.backorder_costs
        )  
        cost = abs(min(self.inventory[0], 0)) * b
        return cost

    def get_holding_costs(self):
        """
        assumes that the same items have same holding costs, even if purchased at different prices, because they take the same amount of space and stock is currently not organized to separate same products.
        assumption is that holding costs equal 25% of the standard purchase price, (per year).

        returns holding costs of current inventory at hand (float)
        """

        cost = max(self.inventory[0], 0) * self.holding_costs
        return cost

    def get_reward(self, action):
        """
        input:
            action: array of shape(1,2) that reflects orders at corresponding prices
            action_memory: dict of actions from beginning of current year, starting with oldest action and ending with current action, keys: days to end of year

        objective is to minimize costs = backlogging + holding costs
            - transportation costs (maybe leave out for now?)
        returns negative reward in self.datetime (current period) (total costs)
        """
        
        purchase_costs = np.dot(self.prices, action) 
        total_costs = (self.get_holding_costs() + self.get_backlogging_costs() + purchase_costs)
        self.acc_purchase_value += purchase_costs
        self.acc_purchase_volume += np.sum(action)
        if self.days_to_end_of_month == 0:
            budget_difference = float(max(-self.budget, 0))
            if self.acc_purchase_value / self.acc_purchase_volume < self.index:  # rebalancing
                total_costs += budget_difference
                if self.budget < 0:
                    total_costs += self.overbudget_punishment
        print('total costs', total_costs)
        return -round(total_costs, 2)
    
    def get_orderdays_to_monthend(self): 
        """ 
        count Tuesdays and Thursdays businessdays left in current month. If 0, current_date is last order date 
        returns integer """
        begin_date=self.current_date
        month_end_date=date(self.current_date.year, self.current_month, calendar.monthrange(self.current_date.year, self.current_date.month)[1])
        last_order_day = np.busday_offset(month_end_date, 0, roll='backward', weekmask = [0,1,0,1,0,0,0])
        busday_count = np.busday_count(begin_date, last_order_day, weekmask=[0,1,0,1,0,0,0], holidays=self.holiday_list)
        return busday_count
        

    def env_start(self):
        """
        The first method called when training starts, called before the
        agent starts.
        Returns:
            The first state from the environment
        """
        
        np.random.seed(0)
        self.budget = 0
        self.accumulated_demand = 0 
        self.last_date=None
        current_date = self.start_date
        self.current_date = np.busday_offset(current_date, 0, roll='forward', weekmask = [0,1,0,1,0,0,0], holidays=self.holiday_list).astype(object) #date = next Tuesday or Thursday 
        self.current_month = current_date.month
        self.days_to_end_of_month = self.get_orderdays_to_monthend() 
        self.demand = self.get_current_demand() #TO DO , is np.array of same len as price array 
        self.update_inventory_level()
        self.budget = 0
        self.acc_purchase_value, self.acc_purchase_volume = 0, 0
        self.done = False
        self.state = (self.inventory, self.budget, self.days_to_end_of_month, self.current_month, self.acc_purchase_value, self.acc_purchase_volume)
        return self.state, self.done

    def env_step(self, action):
        """A step taken by the environment.
        state: (stock, demand)
        new day:
        - change timelog (datetime)
        - then get new stock level after observing (deterministic) arrivals
        - get demand level from data at time self.DATETIME
        - update inventory based on orders taken in the morning
        - observe costs of the day (reward)

        Args:
            action: The action taken by the agent, np.array (1 X num_available_prices )

        Returns:
            (float, state): a tuple of the reward, state.
        """
        
        self.reward = self.get_reward(action) #TO DO check order
        self.update_budget(action)
        self.last_date = self.current_date
        self.current_date = np.busday_offset(self.current_date, 1, roll='forward', weekmask = [0,1,0,1,0,0,0], holidays=self.holiday_list).astype(object)
        self.current_month = self.current_date.month #to account for seasonalities
        self.days_to_end_of_month = self.get_orderdays_to_monthend()
        self.demand = self.get_current_demand() 
        self.update_inventory_level()
        self.inventory[-1] = np.sum(action)
        self.done = self.is_terminal()
        self.state = (self.inventory, self.budget, self.days_to_end_of_month, self.current_month, self.acc_purchase_value, self.acc_purchase_volume)
        return self.reward, self.state, self.done
    

    
    def candidate_budget(self, action):
        """
        updates budget array when demand occurs by buyer 2 (contractual buyer)
        Returns: None

        """

        return self.budget + (
            - np.dot(self.demand, self.prices) + np.dot(self.prices, action)
        )  # budget refers to ordered items (inventory in transit)


    def basestock_level(self): 
        """ optimized basestock quantity, returns integer"""
        ####### TO DO: CHECK ###############
        if self.data is not None: 
            param_list = []
            for i in range(len(self.prices)): 
                f=Fitter(self.data.iloc[:, i], distributions=['gamma'])
                f.fit()
                param_list.append(f.fitted_param['gamma'])
            demand_shape = sum([i[0] for i in param_list])
            demand_scale = sum([i[0] for i in param_list])
        else: 
            demand_shape = np.array((1/8, 1/8, 1/8, 1/8))
            demand_scale = np.array((20, 5, 8, 4))     
        lead_time_plus = len(self.inventory)
        shape_sum_per_price = demand_shape * lead_time_plus
        scale_sum_per_price = demand_scale
        ### approximation Welsh-Satterthwaite
        shape_sum = np.dot(shape_sum_per_price, scale_sum_per_price) ** 2 / np.dot(
            scale_sum_per_price ** 2, shape_sum_per_price
        )
        scale_sum = np.dot(shape_sum_per_price, scale_sum_per_price) / shape_sum
        return int(round(gamma.ppf(self.service_level, shape_sum, scale=scale_sum)))
    
