"""Analytics module for ClimateRisk360.

This module handles data analysis, risk scoring, and model training.
"""
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskAnalyzer:
    """Handles risk analysis and modeling."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the risk analyzer with optional configuration.
        
        Args:
            config: Configuration dictionary with analysis parameters
        """
        self.config = config or {
            'target_variable': 'claims_count',
            'test_size': 0.2,
            'random_state': 42,
            'model_params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'risk_thresholds': {
                'low': 33,
                'medium': 66,
                'high': 100
            }
        }
        
        self.model = None
        self.feature_importances_ = None
        self.metrics_ = {}
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target_col: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
        drop_na: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare data for modeling.
        
        Args:
            df: Input dataframe
            target_col: Name of the target variable column
            feature_cols: List of feature columns to use
            drop_na: Whether to drop rows with missing values
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        if df.empty:
            raise ValueError("Input dataframe is empty")
            
        # Use default target if not specified
        if target_col is None:
            target_col = self.config['target_variable']
            
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # If feature columns not provided, use all numeric columns except target
        if feature_cols is None:
            feature_cols = [
                col for col in df.select_dtypes(include=[np.number]).columns 
                if col != target_col and not col.startswith(('target_', 'actual_'))
            ]
        
        # Filter to selected features and target
        data = df[feature_cols + [target_col]].copy()
        
        # Drop rows with missing values if requested
        if drop_na:
            initial_rows = len(data)
            data = data.dropna(subset=[target_col] + feature_cols)
            if len(data) < initial_rows:
                logger.warning(f"Dropped {initial_rows - len(data)} rows with missing values")
        
        # Separate features and target
        X = data[feature_cols]
        y = data[target_col]
        
        return X, y, feature_cols
    
    def train_model(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        model_params: Optional[Dict] = None
    ) -> Dict:
        """Train a random forest regression model.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_params: Parameters for the random forest model
            
        Returns:
            Dictionary with model metrics
        """
        if model_params is None:
            model_params = self.config['model_params']
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['test_size'],
            random_state=self.config['random_state']
        )
        
        # Initialize and train the model
        self.model = RandomForestRegressor(**model_params)
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train': self._calculate_metrics(y_train, y_pred_train, 'train'),
            'test': self._calculate_metrics(y_test, y_pred_test, 'test')
        }
        
        # Store feature importances
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        self.metrics_ = metrics
        return metrics
    
    def _calculate_metrics(
        self, 
        y_true: Union[pd.Series, np.ndarray], 
        y_pred: Union[pd.Series, np.ndarray],
        prefix: str = ''
    ) -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Add prefix if provided
        if prefix:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
            
        return metrics
    
    def predict_risk(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> Tuple[np.ndarray, Optional[Dict]]:
        """Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            y: Optional true values for calculating metrics
            
        Returns:
            Tuple of (predictions, metrics)
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call train_model() first.")
            
        predictions = self.model.predict(X)
        metrics = None
        
        if y is not None:
            metrics = self._calculate_metrics(y, predictions, 'prediction')
            
        return predictions, metrics
    
    def calculate_risk_categories(self, risk_scores: Union[pd.Series, np.ndarray]) -> pd.Series:
        """Categorize risk scores into low, medium, and high risk."""
        if isinstance(risk_scores, pd.Series):
            risk_scores = risk_scores.values
            
        categories = np.digitize(
            risk_scores, 
            bins=[
                self.config['risk_thresholds']['low'],
                self.config['risk_thresholds']['medium'],
                self.config['risk_thresholds']['high']
            ]
        )
        
        return pd.Series(categories).map({
            0: 'Low',
            1: 'Medium',
            2: 'High',
            3: 'Critical'
        })
    
    def plot_feature_importance(self, top_n: int = 15) -> go.Figure:
        """Plot feature importances."""
        if self.feature_importances_ is None:
            raise ValueError("Feature importances not available. Train the model first.")
            
        # Get top N features
        top_features = self.feature_importances_.head(top_n).sort_values('importance', ascending=True)
        
        # Create bar plot
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=top_features['feature'],
                x=top_features['importance'],
                orientation='h',
                marker_color='#1f77b4'
            )
        )
        
        fig.update_layout(
            title='Feature Importances',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def plot_actual_vs_predicted(
        self, 
        y_true: Union[pd.Series, np.ndarray], 
        y_pred: Union[pd.Series, np.ndarray],
        title: str = 'Actual vs Predicted Values'
    ) -> go.Figure:
        """Create a scatter plot of actual vs predicted values."""
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(color='#1f77b4')
            )
        )
        
        # Add reference line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            showlegend=True,
            width=800,
            height=600
        )
        
        return fig
    
    def plot_risk_trends(
        self, 
        df: pd.DataFrame, 
        region_col: str = 'region',
        date_col: str = 'date',
        risk_col: str = 'risk_score'
    ) -> go.Figure:
        """Plot risk trends over time by region."""
        if not all(col in df.columns for col in [region_col, date_col, risk_col]):
            raise ValueError("Required columns not found in dataframe")
            
        fig = px.line(
            df, 
            x=date_col, 
            y=risk_col, 
            color=region_col,
            title='Risk Score Trends by Region',
            labels={risk_col: 'Risk Score', date_col: 'Date'}
        )
        
        # Add risk threshold lines
        thresholds = self.config['risk_thresholds']
        for level, value in thresholds.items():
            fig.add_hline(
                y=value,
                line_dash='dash',
                line_color='red' if level != 'low' else 'green',
                annotation_text=f"{level.capitalize()} Risk Threshold",
                annotation_position='bottom right'
            )
        
        fig.update_layout(
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model has been trained yet")
            
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'RiskAnalyzer':
        """Load a trained model from disk."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        analyzer = cls()
        analyzer.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return analyzer
