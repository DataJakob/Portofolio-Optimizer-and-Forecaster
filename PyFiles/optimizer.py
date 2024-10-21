import numpy as np

from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize

import matplotlib.pyplot as plt

class Optimized:

    def __init__(self, stocks, portofolio_cloud, 
                 tot_cov_mat, neg_cov_mat, 
                 returns, mean, stds
                 ):
        
        self.stocks = stocks
        self.portofolio_cloud = portofolio_cloud
        self.tot_cov_mat = tot_cov_mat
        self.neg_cov_mat = neg_cov_mat
        self.returns = returns
        self.mean = mean
        self.stds = stds


    def PortofolioOptimizer(self):


        c1 = Bounds(0,1)

        #Creating arbitrary values for weigths
        c2 = LinearConstraint(np.ones((len(self.stocks),), dtype=int),1,1)

        #Creating arbitrary values for the weigths
        weigths = np.ones(len(self.stocks))
        #And making them add up to 100%.
        decVar = weigths/np.sum(weigths)


        opt_port = []
        opt_spec = ['Minimum Variance', 'Highest Sortino', 'Highest Sharpe']
        special_params = [self.tot_cov_mat, self.tot_cov_mat, self.neg_cov_mat]
        for i in range(0,3,1):
            if  i ==0:  
                Z = lambda w: np.sqrt(w@special_params[i]@w.T)
            else:
                Z = lambda w: np.sqrt(w@special_params[i]@w.T)/(self.mean@w)

            res = minimize(Z, decVar, method='trust-constr', constraints = c2, bounds = c1)
            w = res.x
            ret = sum(w*self.mean)
            risk = (w@special_params[0]@w.T)**.5
            opt_port.append([w, ret, risk, opt_spec[i]])
        self.optimized_portofolios = opt_port


    def plot_optimization(self):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.scatter(y=self.portofolio_cloud['Return'], 
                   x=self.portofolio_cloud['Total_risk'], 
                   c=self.portofolio_cloud['Negative_risk'],
                    marker='o', alpha=.3)

        for i in range(3):
            ax.scatter(x=self.optimized_portofolios[i][2],
                       y=self.optimized_portofolios[i][1], 
                       label=self.optimized_portofolios[i][3], s=105)

        ax.scatter(y=self.mean, x= self.stds,color='brown')
        for i in range(len(self.stocks)):
            plt.annotate(self.stocks[i], (self.stds[i]+0.0005,self.mean[i]))
        plt.xlabel('Risk')
        plt.ylabel('Return')
        plt.legend()
        plt.title('Portofolio Optimzation Plot', fontsize=20)
        # plt.show()
        plt.savefig('optimization.png')
