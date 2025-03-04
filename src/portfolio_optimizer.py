import pandas as pd
import numpy as np
import cvxpy as cp
import logging

logging.basicConfig(
    filename='../logs/portfolio_optimizer.log',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logging.info('Logging started for Portfolio Optimizer module')


class PortfolioOptimizer:
    """
    Class to optimize a portfolio based on forecasted returns.

    Attributes:
        forecasts (pd.DataFrame): Forecasted daily prices for assets (e.g., TSLA, BND, SPY).
        returns (pd.DataFrame): Daily returns computed from forecasts.
    """

    def __init__(self, forecasts):
        self.forecasts = forecasts
        self.returns = self.forecasts.pct_change().dropna()
        logging.info("PortfolioOptimizer instance created with assets: %s",
                     self.forecasts.columns.tolist())

    def compute_annual_metrics(self):
        """Computes annualized return and covariance matrix."""
        logging.info("Computing annual metrics")
        try:
            annual_returns = (1 + self.returns.mean()) ** 252 - \
                1  # Compound daily to annual
            cov_matrix = self.returns.cov() * 252  # Annualized covariance
            logging.info("Annual returns: %s", annual_returns.to_dict())
            logging.info("Covariance matrix computed")
            return annual_returns, cov_matrix
        except Exception as e:
            logging.error("Error computing annual metrics: %s", e)
            raise

    def optimize_sharpe(self, risk_free_rate=0.02):
        """Optimizes portfolio weights to maximize Sharpe Ratio."""
        logging.info("Starting Sharpe Ratio optimization")
        try:
            annual_returns, cov_matrix = self.compute_annual_metrics()
            n_assets = len(self.forecasts.columns)
            weights = cp.Variable(n_assets)
            portfolio_return = annual_returns.values @ weights
            portfolio_risk = cp.quad_form(weights, cov_matrix.values)
            objective = cp.Maximize(portfolio_return - risk_free_rate)
            # Normalize risk, long-only
            constraints = [portfolio_risk <= 1, weights >= 0]
            problem = cp.Problem(objective, constraints)
            problem.solve()
            optimal_weights = weights.value
            if optimal_weights is None:
                raise ValueError("Optimization failed to converge")
            # Normalize weights to sum to 1
            optimal_weights = optimal_weights / np.sum(optimal_weights)
            logging.info("Optimal weights (normalized): %s", optimal_weights)
            return pd.Series(optimal_weights, index=self.forecasts.columns)
        except Exception as e:
            logging.error("Error in Sharpe optimization: %s", e)
            raise

    def portfolio_performance(self, weights, risk_free_rate=0.02):
        """Computes portfolio return, volatility, Sharpe Ratio, and VaR."""
        logging.info("Computing portfolio performance")
        try:
            annual_returns, cov_matrix = self.compute_annual_metrics()
            portfolio_return = weights @ annual_returns
            portfolio_volatility = np.sqrt(weights @ cov_matrix @ weights)
            sharpe_ratio = (portfolio_return - risk_free_rate) / \
                portfolio_volatility
            portfolio_returns = (self.returns * weights).sum(axis=1)
            var_95 = np.percentile(portfolio_returns, 5) * \
                np.sqrt(252)  # Annualized VaR at 95%
            logging.info("Portfolio metrics - Return: %.4f, Volatility: %.4f, Sharpe: %.4f, VaR: %.4f",
                         portfolio_return, portfolio_volatility, sharpe_ratio, var_95)
            return {
                'Return': portfolio_return,
                'Volatility': portfolio_volatility,
                'Sharpe Ratio': sharpe_ratio,
                'VaR (95%)': var_95
            }, portfolio_returns
        except Exception as e:
            logging.error("Error computing portfolio performance: %s", e)
            raise
