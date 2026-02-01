"""
Classification module.

Predicts sector outperformance using ML models.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


def create_binary_target(
    panel: pd.DataFrame,
    car_column: str = 'car_60m',
    threshold: float = 0.0,
    method: str = 'absolute'
) -> pd.Series:
    """
    Create binary classification target.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel with CAR
    car_column : str
        CAR column to use
    threshold : float
        Threshold for positive class
    method : str
        'absolute' (car > threshold) or 'relative' (car > median)
        
    Returns
    -------
    pd.Series
        Binary target (1 = outperform)
    """
    car = panel[car_column]
    
    if method == 'absolute':
        target = (car > threshold).astype(int)
    elif method == 'relative':
        target = (car > car.median()).astype(int)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Created binary target: {target.mean():.1%} positive")
    
    return target


def prepare_classification_features(
    panel: pd.DataFrame,
    feature_cols: List[str] = None,
    categorical_cols: List[str] = None
) -> pd.DataFrame:
    """
    Prepare features for classification.
    
    Parameters
    ----------
    panel : pd.DataFrame
        Event panel
    feature_cols : list, optional
        Feature columns to use
    categorical_cols : list, optional
        Categorical columns to encode
        
    Returns
    -------
    pd.DataFrame
        Feature matrix
    """
    if feature_cols is None:
        # Default features
        feature_cols = [
            'cumulative_attention', 'importance_weight',
            'pre_volatility', 'pre_return',
            'sentiment_compound', 'fiscal_intensity',
            'certainty_score', 'actionability_score'
        ]
    
    # Filter to available columns
    available = [c for c in feature_cols if c in panel.columns]
    
    X = panel[available].copy()
    
    # Encode categoricals
    if categorical_cols:
        for col in categorical_cols:
            if col in panel.columns:
                dummies = pd.get_dummies(panel[col], prefix=col, drop_first=True)
                X = pd.concat([X, dummies], axis=1)
    
    # Fill missing with median
    X = X.fillna(X.median())
    
    logger.info(f"Prepared {X.shape[1]} features for classification")
    
    return X


def train_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'lightgbm',
    cv_folds: int = 5,
    random_state: int = 42
) -> Tuple[object, Dict]:
    """
    Train a classification model.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Binary target
    model_type : str
        Model type: 'lightgbm', 'logistic', or 'random_forest'
    cv_folds : int
        Number of CV folds
    random_state : int
        Random seed
        
    Returns
    -------
    tuple
        (trained_model, cv_metrics)
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    # Handle missing
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X[mask]
    y_clean = y[mask]
    
    if len(y_clean) < cv_folds * 2:
        logger.warning("Insufficient data for cross-validation")
        return None, {'error': 'Insufficient data'}
    
    # Choose model
    if model_type == 'lightgbm':
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
                verbose=-1
            )
        except ImportError:
            logger.warning("LightGBM not installed, using logistic regression")
            model_type = 'logistic'
    
    if model_type == 'logistic':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state
        )
    
    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=random_state
        )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    auc_scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring='roc_auc')
    acc_scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring='accuracy')
    
    metrics = {
        'model_type': model_type,
        'cv_auc_mean': auc_scores.mean(),
        'cv_auc_std': auc_scores.std(),
        'cv_accuracy_mean': acc_scores.mean(),
        'cv_accuracy_std': acc_scores.std(),
        'n_samples': len(y_clean),
        'n_features': X_clean.shape[1],
        'positive_rate': y_clean.mean()
    }
    
    logger.info(f"CV AUC: {metrics['cv_auc_mean']:.3f} ± {metrics['cv_auc_std']:.3f}")
    
    # Train final model
    model.fit(X_clean, y_clean)
    
    return model, metrics


def get_feature_importance(
    model,
    feature_names: List[str],
    importance_type: str = 'split'
) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    Parameters
    ----------
    model : trained model
        Trained classifier
    feature_names : list
        Feature names
    importance_type : str
        Type of importance
        
    Returns
    -------
    pd.DataFrame
        Feature importance
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    importance_df['importance_pct'] = importance_df['importance'] / importance_df['importance'].sum() * 100
    
    return importance_df


def predict_outperformance(
    model,
    X_new: pd.DataFrame
) -> pd.DataFrame:
    """
    Predict outperformance probability for new data.
    
    Parameters
    ----------
    model : trained model
        Trained classifier
    X_new : pd.DataFrame
        New feature data
        
    Returns
    -------
    pd.DataFrame
        Predictions with probabilities
    """
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_new)[:, 1]
    else:
        probs = model.predict(X_new)
    
    predictions = pd.DataFrame({
        'prediction': (probs > 0.5).astype(int),
        'probability': probs
    }, index=X_new.index)
    
    return predictions
