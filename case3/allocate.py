import numpy as np
import pandas as pd
import scipy as sp


# Implementing Hierarchical Risk Parity

k = -1
counter = 0

window = 252*18

df = pd.read_csv('Training Data_Case 3.csv', index_col=0)
daily_returns = df.pct_change().iloc[1:, :]


def allocate_portfolio(asset_prices):
    global df, daily_returns, k, counter, window

    df = pd.concat([df.iloc[-window:, :], pd.Series(asset_prices, index=df.columns).to_frame().T], axis=0)
    
    daily_returns = df.pct_change()


    # Step 1: Calculate the Covariance Matrix
    cov_matrix = daily_returns.cov()

    if k == -1:
        k_values = range(1,10)

        best_sharpe_ratio = -np.inf
        best_k = None
        best_weights = None
        for k in k_values:
            hrp_weights, portfolio_sharpe_ratio, portfolio_returns = hrp_optimizer(cov_matrix, k, daily_returns)
            if portfolio_sharpe_ratio > best_sharpe_ratio:
                best_sharpe_ratio = portfolio_sharpe_ratio
                best_k = k
        k = best_k
    
    w, _, _ = hrp_optimizer(cov_matrix, k, daily_returns)

    print(counter)
    counter += 1
    return w
    

def hrp_optimizer(covariance_matrix, k, returns_df):
    # Step 1: Calculate the Correlation Matrix
    correlation_matrix = covariance_matrix.corr()
    
    # Step 2: Calculate the Distance Matrix
    distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix))
    
    # Step 3: Calculate the Linkage Matrix
    linkage_matrix = sp.cluster.hierarchy.linkage(distance_matrix, 'centroid')
    
    # Step 4: Calculate the Cluster Assignments
    cluster_assignments = sp.cluster.hierarchy.fcluster(linkage_matrix, k, criterion='inconsistent')
    
    # Step 5: Calculate the Inverse Variance Portfolios
    ivp = 1 / np.diag(covariance_matrix)
    ivp /= ivp.sum()
    
    # Step 6: Calculate the Hierarchical Risk Parity Portfolio Weights
    hrp_weights = pd.Series(index=covariance_matrix.columns, dtype=np.float64)
    for cluster_id in np.unique(cluster_assignments):
        cluster_indices = np.where(cluster_assignments == cluster_id)[0]
        cluster_covariance = covariance_matrix.iloc[cluster_indices, cluster_indices]
        cluster_ivp = ivp[cluster_indices]
        cluster_weight = cluster_ivp / np.sum(cluster_ivp)
        hrp_weights[cluster_covariance.index] = cluster_weight
    hrp_weights /= hrp_weights.sum()

    # Step 7: Calculate the Portfolio Statistics
    portfolio_returns = (hrp_weights * returns_df.mean()).sum()
    portfolio_volatility = np.sqrt(np.dot(hrp_weights.T, np.dot(covariance_matrix, hrp_weights)))
    portfolio_sharpe_ratio = portfolio_returns / portfolio_volatility
    
    return hrp_weights, portfolio_sharpe_ratio, portfolio_returns




def grading(testing): #testing is a pandas dataframe with price data, index and column names don't matter
    weights = np.full(shape=(len(testing.index),10), fill_value=0.0)
    for i in range(0,len(testing)):
        unnormed = np.array(allocate_portfolio(list(testing.iloc[i,:])))
        positive = np.absolute(unnormed)
        normed = positive/np.sum(positive)
        weights[i]=list(normed)
    capital = [1]
    for i in range(len(testing) - 1):
        shares = capital[-1] * np.array(weights[i]) / np.array(testing.iloc[i,:])
        capital.append(float(np.matmul(np.reshape(shares, (1,10)),np.array(testing.iloc[i+1,:]))))
    returns = (np.array(capital[1:]) - np.array(capital[:-1]))/np.array(capital[:-1])
    return np.mean(returns)/ np.std(returns) * (252 ** 0.5), capital, weights


