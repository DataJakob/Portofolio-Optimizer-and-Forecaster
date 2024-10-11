from datetime import datetime as datetime
import yfinance as yf 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PortofolioCloud:

    def __init__(self, stocks, 
                 cloud_size, 
                 start_date, end_date=None
                 ):
        self.stocks = stocks
        self.cloud_size = cloud_size
        self.start_date = start_date
        if end_date != None:
            self.end_date = end_date
        else: 
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.stocks_matrix = None
        self.mean = None
        self.std = None
        self.neg_cov_mat = None
        self.tot_cov_mat = None
        self.cloud = None
        self.returns = None
        self.neg_risk = None
        self.tot_risk = None
        self.df = None

    def dataRetriever(self):
        df = []
        for i in range(0,len(self.stocks),1):
            yahoo = yf.download(self.stocks[i], start=self.start_date, end=self.end_date)
            data = [0]
            for j in range(0, len(yahoo['Adj Close'])-1,1):
                selecive_data = ((yahoo['Adj Close'][j+1]/yahoo['Adj Close'][j])-1)
                data.append(selecive_data)
            adj_close = yahoo['Adj Close']
            relevant  = [self.stocks[i], data, adj_close]
            df.append(relevant)
        self.stocks_matrix = df

    def means(self):
        means = []
        for i in range(0,len(self.stocks_matrix),1):
            means.append(pd.Series(self.stocks_matrix[i][1]).mean())
        self.mean = np.array(means)


    def stds(self):
        means = []
        for i in range(0,len(self.stocks_matrix),1):
            means.append(pd.Series(self.stocks_matrix[i][1]).std())
        self.std = np.array(means)

    def cov_matrix(self):
        total_cov_matrix = pd.DataFrame()
        negative_cov_matrix =  pd.DataFrame()
        for i in range(0, len(self.stocks_matrix),1):
            total_cov_matrix[self.stocks_matrix[i][0]] = self.stocks_matrix[i][1]
            stock = pd.Series(self.stocks_matrix[i][1])
            neg = stock.loc[stock < 0]
            negative_cov_matrix[self.stocks_matrix[i][0]] = neg
        self.neg_cov_mat =  negative_cov_matrix.cov()
        self.tot_cov_mat = total_cov_matrix.cov()

    def cloud_generator(self):
        portofolios = []
        for i in range(0,self.cloud_size,1):
            single = np.random.dirichlet(np.ones(len(self.stocks)),size=1)
            portofolios.append(single)
        self.cloud = portofolios

    def port_returns(self):
        po_returns = []
        for i in range(0, len(self.cloud),1):
            po_returns.append(sum(self.mean@self.cloud[i].T))
        self.returns = po_returns


    def risks(self):
        def portofolio_std(ps, cov):
            #ps is short for portofolios
            pstd = [(ps[i] @ cov @ ps[i].T)**.5 for i in range(len(ps))]
            portofolio_std = pd.Series([pstd[i][0][0] for i in range(len(ps))])

            return portofolio_std
        self.neg_risk = portofolio_std(self.cloud, self.neg_cov_mat)
        self.tot_risk = portofolio_std(self.cloud, self.tot_cov_mat)

    def final_df(self):
        self.dataRetriever()
        self.means()
        self.stds()
        self.cov_matrix()
        self.cloud_generator()
        self.port_returns()
        self.risks()

        df = pd.DataFrame()
        colnames =  ['Weigths', 'Return', 'Negative_risk','Total_risk']
        colvalues = [self.cloud, self.returns, self.neg_risk, self.tot_risk]
        
        for i in range(0,len(colnames),1):
            df[colnames[i]] = colvalues[i]
        df['Sharpe_Ratio'] = df['Return']/(df['Total_risk'])
        df['Sortino_Ratio'] =df['Return']/(df['Negative_risk']+1)
        df = df.loc[df['Return'] >= 0].reset_index()
        del df['index']

        self.df = df

    def plot_stocks(self):
        fig, ax = plt.subplots()
        for i in range(len(self.stocks_matrix)):
            ax.plot(self.stocks_matrix[i][2]/self.stocks_matrix[i][2][0], label=self.stocks[i])
        plt.legend()
        plt.show()

        




