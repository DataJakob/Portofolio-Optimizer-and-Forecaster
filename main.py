# Import modules
from PyFiles.portofolioCloud import PortofolioCloud
from PyFiles.optimizer import Optimized
from PyFiles.forecaster import Forecaster



"""
--------------------
State input variables and generate a portfolio cloud
"""
stocks = ['AKRBP.OL','BORR.OL','EPR.OL','EQNR.OL','LSG.OL','MOWI.OL','SRBNK.OL','SATS.OL']
#format: year-month-day
startPor = '2020-02-01'
hold_portofolio_days = 21
portofolio_Distributions = 5000
alfa = PortofolioCloud(stocks,500,startPor)
alfa.final_df()
alfa.plot_stocks()

"""
-----------------------------
Optimize the portofolio in regards to:
 minimum variance, lowest sharpe ratio and lowest sortino ratio
"""
bravo = Optimized(alfa.stocks, alfa.df,
                  alfa.tot_cov_mat,alfa.neg_cov_mat,
                  alfa.returns, alfa.mean, alfa.std)
bravo.PortofolioOptimizer()
bravo.plot_optimization()


"""
--------------------------
Forecast a desired portofolio using a MME (MLR, RFR, RFR)
Not assuming a seasonal component
"""
# 0 = MinVar | 1 = Sortino | 2 = Sharpe
mychoice = 1
charlie = Forecaster(stocks=alfa.stocks_matrix,
                     optimized_portofolios=bravo.optimized_portofolios,
                     choosen_portofolio=mychoice,
                     forecast_period=20)
charlie.show_distribution()
charlie.forecast()