import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class AutoGARCH:
    def __init__(self, p_max=3, q_max=3):
        self.p_max = p_max
        self.q_max = q_max
        self.best_result = None
        self.split_index = 0
        self.rescale_factor = 1.0

    def fit(self, full_returns, test_start_index):
        """
        Fits GARCH on data up to test_start_index.
        """
        self.split_index = test_start_index
        
        # Scaling Check: GARCH fails if returns are tiny (like 0.0005). 
        # We multiply by 100 to help convergence, then divide back later.
        if full_returns.abs().mean() < 0.1:
            print("   (Scaling returns by 100 for stability...)")
            self.rescale_factor = 100.0
        
        scaled_returns = full_returns * self.rescale_factor
        
        best_bic = float('inf')
        best_params = {}

        print(f"AutoGARCH: Tuning (p=1-{self.p_max}, q=1-{self.q_max})...")

        # Grid Search
        for p in range(1, self.p_max + 1):
            for q in range(1, self.q_max + 1):
                try:
                    # 'last_obs' tells it where the training data ends.
                    # It ignores data after this index for fitting parameters.
                    model = arch_model(scaled_returns, vol='Garch', p=p, q=q, dist='Normal', rescale=False)
                    res = model.fit(last_obs=test_start_index, disp='off')
                    
                    if res.bic < best_bic:
                        best_bic = res.bic
                        self.best_result = res
                        best_params = {'p': p, 'q': q}
                except:
                    continue

        print(f"Best Model: GARCH({best_params.get('p')}, {best_params.get('q')}) | BIC: {best_bic:.2f}")
        return self.best_result

    def evaluate(self, y_test_actual, horizon=5):
        """
        Forecasts the test set and calculates metrics vs y_test_actual.
        """
        if self.best_result is None:
            raise ValueError("Run .fit() first!")

        print("Generating GARCH Forecasts...")
        
        # 1. Forecast starting exactly where training ended
        # horizon=5 implies we predict variance for t+1, t+2... t+5
        forecasts = self.best_result.forecast(start=self.split_index, horizon=horizon, reindex=False)
        
        # 2. Extract Variance (dimensions: [N_test, 5])
        var_preds = forecasts.variance.values
        
        # 3. Sum variances over 5 minutes (standard statistical rule)
        five_min_variance = np.sum(var_preds, axis=1)
        
        # 4. Convert to Volatility (Sqrt) and Re-scale
        # We divide by rescale_factor because we multiplied inputs by 100 earlier
        pred_vol = np.sqrt(five_min_variance) / self.rescale_factor
        
        # 5. Align dimensions
        # Sometimes forecast produces 1 extra or 1 less row depending on indices
        min_len = min(len(pred_vol), len(y_test_actual))
        preds = pred_vol[:min_len]
        actuals = y_test_actual[:min_len]
        
        # 6. Metrics
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mae = mean_absolute_error(actuals, preds)
        r2 = r2_score(actuals, preds)
        
        print(f"\nGARCH RESULTS (Test Set):")
        print(f"   RMSE: {rmse:.6f}")
        print(f"   MAE:  {mae:.6f}")
        print(f"   R^2:  {r2:.6f}")
        
        return preds, {"RMSE": rmse, "MAE": mae, "R2": r2}