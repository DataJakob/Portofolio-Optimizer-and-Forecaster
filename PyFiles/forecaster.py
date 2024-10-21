import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Statistics libraries
import scipy.stats as st
import statsmodels.formula.api as smf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.deterministic import DeterministicProcess
from sklearn.linear_model import LinearRegression


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_absolute_error as MAE

class Forecaster:

    def __init__(
            self,
            stocks=None,    #alfa.stocks_matrix
            optimized_portofolios = None,
            choosen_portofolio = None,
            forecast_period = None
            ):
        self.stocks = stocks
        self.optimized_portofolios = optimized_portofolios
        self.choosen_portofolio = choosen_portofolio
        self.portofolio_movement = None
        self.forecast_period = forecast_period
        self.predicted_forecast = None

    def show_distribution(self):
        best_por = self.optimized_portofolios[self.choosen_portofolio]
        print(self.optimized_portofolios[self.choosen_portofolio][3])
        print('Stock distributions:')
        print(pd.DataFrame([round(best_por[0][i],2) for i in range(0,len(self.stocks),1)], 
                     index=[self.stocks[i][0] for i in range(0, len(self.stocks),1)]).transpose())
        time_cum = []
        for i in range(0, len(self.stocks[0][1])):
            day_cum = []
            day_sel = 0

            for j in range(0,len(best_por),1):
                day_sel += best_por[0][j]*self.stocks[j][1][i]
                if j == (len(best_por)-1):
                    
                    day_cum.append(day_sel+1)
            time_cum.append(day_cum[0])
        self.portofolio_movement = pd.Series(np.cumprod(time_cum))

    def forecast(self):
        #Creating X and y ffeature for capturing the trend
        y = self.portofolio_movement.copy()
        dp = DeterministicProcess(index=y.index,
                                constant=True,
                                order=2,
                                drop=True)
        X = dp.in_sample()


        # Splitting the data into different sets
        def tts(X,y):
            idx_train, idx_test = train_test_split(y.index,
                                                test_size=60,
                                                random_state=42,
                                                shuffle=False)
            X_train, y_train = X.loc[idx_train,:], y.loc[idx_train]
            X_test, y_test = X.loc[idx_test,:], y.loc[idx_test]
            return [X_train,y_train, X_test,y_test]
        first_split = tts(X, self.portofolio_movement)
        X_train, y_train = first_split[0],first_split[1]
        X_test, y_test = first_split[2], first_split[3]
        fit_idx = np.linspace(0, len(X_train), len(X_train))
        pred_idx = np.linspace(len(y_train)+1, len(y_train)+len(X_test),len(X_test))


        # Fitting a linear regression model to capture trend
        model_trend = LinearRegression()
        model_trend.fit(X_train, y_train)

        # Predict the train and test set
        y_fit_trend = pd.Series(model_trend.predict(X_train),
                        index=y_train.index)

        y_pred_trend = pd.Series(model_trend.predict(X_test), 
                            index=y_test.index)
        

        #Generating different models
        def grid_searcher(depth_int='', nest_int=''):
            models = []
            for i in range(0,len(depth_int),1):
                for j in range(0,len(nest_int),1):
                    sel_model = RandomForestRegressor(random_state=42,
                                                    max_depth=int(depth_int[i]),
                                                    n_estimators=int(nest_int[j]))
                    models.append(sel_model)
            return models


        #Running cross_validation on time series
        def timeSeriesCV(model='', train='', test='', cv=''):
            baseline = 0.5
            baselen = len(X_train)
            split = 0.5/(cv+1)
            scores = []
            for i in range(0,cv,1):
                ref_idx_train = int(baselen*((split*i)+baseline))
                ref_idx_test = int(baselen+((split*i+1)+baseline))
                sel_X_train = train.iloc[:ref_idx_train]
                sel_y_train = test.iloc[:ref_idx_train]
                sel_X_test = train.iloc[ref_idx_train: ref_idx_test]
                sel_y_test = test.iloc[ref_idx_train: ref_idx_test]
                model.fit(sel_X_train,sel_y_train)
                test_pred = model.predict(sel_X_test)
                scores.append(MAPE(test_pred,sel_y_test))
            return np.mean(scores)


        #Finding best model
        def best_model(depth_int, nest_int, train, test, cv):
            models =  grid_searcher(depth_int, nest_int)
            results = []
            for i in range(0,len(models),1):
                score = timeSeriesCV(model=models[i], train=train, test=test, cv=cv)
                model = models[i]
                results.append([score,model])
            low = min([results[i][0] for i in range(len(results))])
            idx = np.where([results[i][0] for i in range(len(results))]== low)
            best_model = results[int(idx[0])][1]
            return best_model
            # return results

        def generate_lags(data, n_lags):
            df = pd.DataFrame()
            for i in range(1,n_lags+1,1):
                df['lag'+str(i)] = data.shift(i, fill_value=1)
            return df
        def normalize(arr):
            answer = (arr-arr.mean())/arr.std()
            return answer
        
        # Define new explanatory variables
        X_train_cyc = generate_lags(y_train,1)
        X_test_cyc = generate_lags(y_test,1)
        # Define new target variable
        y_train_cyc = (y_train - y_fit_trend)
        y_test_cyc = (y_test - y_pred_trend)
        # Model fitting and  prediction
        model_cyc = best_model(np.linspace(4,12,8),  np.linspace(5,13,9), X_train_cyc, y_train_cyc, cv=7)
        model_cyc.fit(X_train_cyc,y_train_cyc)
        y_fit_cyc = model_cyc.predict(X_train_cyc)
        y_pred_cyc = model_cyc.predict(X_test_cyc)



        # Define new target variable
        y_train_resid = y_train - y_fit_trend - y_fit_cyc
        y_test_resid = y_test - y_pred_trend -  y_pred_cyc
        # Define new explanatory variables
        X_train_resid = pd.concat([X_train, X_train_cyc],axis=1)
        X_test_resid = pd.concat([X_test, X_test_cyc], axis=1)
        # Model fitting and  prediction
        model_resid = best_model(np.linspace(4,12,8),  np.linspace(5,13,9), X_train_resid, y_train_resid, cv=7)
        model_resid.fit(X_train_resid,y_train_resid)
        y_fit_resid = model_resid.predict(X_train_resid)
        y_pred_resid = model_resid.predict(X_test_resid)


        final_fit  = y_fit_trend + y_fit_cyc + y_fit_resid
        final_pred = y_pred_trend.reset_index()[0] + y_pred_cyc + y_pred_resid
        final_fit_mae = round(MAPE(final_fit, y_train),2)
        final_pred_mae = round(MAPE(final_pred, y_test),2)
        # Multimodel function for forecasting values
        def multi_model(pred):
            new_y1 = model_trend.predict(pred.loc[:,: 'trend_squared'])
            new_y2 = model_cyc.predict(pred[['lag1']])
            new_y3 = model_resid.predict(pred)
            new_forecast = new_y1[0] + new_y2[0] + new_y3[0]
            return new_forecast
        # Forecast function that uses a Recursive strategy
        def recursive():
            forecast_features = pd.DataFrame(columns= ['const', 'trend', 'trend_squared', 'lag1'], index=[0])
            previous_row = X_test_resid.iloc[-1:].reset_index()
            forecasts = []
            for i in range(0,  self.forecast_period,1):
                if i == 0:
                    previous_target = y_test[-1:].reset_index()[0][0]
                else:
                    previous_target = forecasts[-1]
                new_row = [1, 
                        previous_row['trend'][0]+i,
                        (previous_row['trend'][0]+i)**2,
                        previous_target]
                new_pred = pd.DataFrame({'const':new_row[0],
                                        'trend':new_row[1],
                                        'trend_squared':new_row[2],
                                        'lag1':new_row[3]}, 
                                        index=[i])
                forecasts.append(multi_model(new_pred))
                forecast_features.loc[i] = new_row
            return [forecasts, forecast_features]
        forecasts = recursive()[0]
        for_idx = np.linspace(len(y)+1, len(y)+len(forecasts),self.forecast_period)
        self.predicted_forecast = forecasts 


        pam = self.portofolio_movement
        fig = plt.figure(figsize=(10,10))

        ax0 = plt.subplot2grid((5,3),(0,0), colspan=3, rowspan=1)
        ax0.plot(pam, color='black', alpha=.3)

        ax1 = plt.subplot2grid((5,3),(1,0)) 
        ax1.plot(pam,color='black', alpha=.3)
        ax1.plot(y_fit_trend, color='#0CCE6B')
        ax1.plot(pred_idx, y_pred_trend, color='#ED7D3A')

        ax2 = plt.subplot2grid((5,3),(1,1))
        ax2.plot(normalize(y_train_cyc), color='black', alpha=.3)
        ax2.plot(normalize(y_test_cyc), color='black', alpha=.3)
        ax2.plot(normalize(y_fit_cyc), color='#0CCE6B')
        ax2.plot(pred_idx, normalize(y_pred_cyc), color='#ED7D3A')


        ax3 = plt.subplot2grid((5,3),(1,2)) 
        ax3.plot(normalize(y_train_resid), color='black', alpha=.3)
        ax3.plot(normalize(y_test_resid), color='black', alpha=.3)
        ax3.plot(normalize(y_fit_resid), color='#0CCE6B')
        ax3.plot(pred_idx, normalize(y_pred_resid), color='#ED7D3A')

        ax4 = plt.subplot2grid((5,3),(2,0), colspan=2, rowspan=2)
        ax4.plot(pam, color='black', alpha=.3)
        ax4.plot(final_fit, color='#EF2D56', label='Fit MAPE: '+str(final_fit_mae))
        ax4.plot(pred_idx, final_pred, color='orange', label='Pred MAPE: '+str(final_pred_mae))
        plt.legend()


        ax5 = plt.subplot2grid((5,3),(2,2), rowspan=2)
        ax5.plot(pam, color='black', alpha=.3)
        ax5.plot(for_idx, forecasts, color='black')
        ax5.plot(pred_idx, final_pred, color='#ED7D3A')
        ax5.set_xlim(len(X_train)-50)

        axes = [ax0,ax1,ax2,ax3,ax4,ax5]
        axes_names = ['Portofolio movement','Trend','Cycles','Residuals', 'Model performance','Forecast']
        for i in range(len(axes)):
            axes[i].set_title(axes_names[i])
            axes[i].set_xlabel('Time in days')
            axes[i].set_ylabel('Return in %')

        fig.suptitle('ML forecasting of optimized portofolio', fontsize=20)

        fig.tight_layout()
        # plt.show()
        plt.savefig('forecast.png')
