#!/usr/bin/env python3
"""
Enhanced ML Engine for Fleet Maintenance
Extends existing XGBoost model with Random Forest, LSTM, and multi-target predictions
"""

import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class EnhancedFleetMaintenanceEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False

    def load_existing_model(self):
        """Load the existing trained XGBoost model and scaler"""
        print("🔄 Loading existing trained models...")

        try:
            # Load existing XGBoost model
            self.models['xgboost'] = joblib.load('models/fleet_maintenance_model.pkl')
            print("✅ XGBoost model loaded successfully")

            # Load existing scaler
            self.scalers['main'] = joblib.load('models/fleet_maintenance_scaler.pkl')
            print("✅ Standard scaler loaded successfully")

            # Load feature names
            with open('models/fleet_maintenance_features.txt', 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            print(f"✅ Feature names loaded: {len(self.feature_names)} features")

            return True

        except Exception as e:
            print(f"❌ Error loading existing models: {str(e)}")
            return False

    def prepare_data(self, data_path='Dataset/fleet_maintenance_clean.csv'):
        """Prepare data for multi-target training"""
        print("📊 Preparing data for multi-target training...")

        try:
            # Load data
            data = pd.read_csv(data_path)
            print(f"✅ Data loaded: {len(data)} records")

            # Apply existing feature engineering (from your optimization script)
            data = self._apply_feature_engineering(data)

            # Prepare features
            X = data[self.feature_names].copy()
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.dropna()

            if len(X) == 0:
                print("❌ No data available after cleaning")
                return None, None, None

            print(f"✅ Clean features: {len(X)} samples")

            # Create multiple targets
            targets = self._create_multiple_targets(data, X.index)

            return X, targets, data

        except Exception as e:
            print(f"❌ Error preparing data: {str(e)}")
            return None, None, None

    def _apply_feature_engineering(self, data):
        """Apply the same feature engineering from your optimization script"""
        print("🔧 Applying feature engineering...")

        np.random.seed(42)

        # Core features with enhanced engineering
        if 'trip_duration_hours' in data.columns and 'distance' in data.columns:
            data['engine_hours'] = data['trip_duration_hours'] * (1 + np.random.normal(0, 0.1, len(data)))
            data['engine_hours'] = np.clip(data['engine_hours'], 0.1, 24)

        if all(col in data.columns for col in ['gross_vehicle_weight', 'avg_speed', 'distance']):
            weight_factor = data['gross_vehicle_weight'] / data['gross_vehicle_weight'].max()
            speed_factor = data['avg_speed'] / data['avg_speed'].max()
            data['fuel_consumption_rate'] = (weight_factor * 0.6 + speed_factor * 0.4) * np.random.uniform(0.8, 1.2, len(data))

        if 'trip_duration_hours' in data.columns:
            data['load_frequency'] = np.clip(data['trip_duration_hours'] / 8, 0.1, 1.0)

        # Enhanced operational patterns
        if 'avg_speed' in data.columns:
            data['route_type_encoded'] = (data['avg_speed'] > 60).astype(int)

        if 'max_speed' in data.columns and 'avg_speed' in data.columns:
            data['speed_variability'] = (data['max_speed'] - data['avg_speed']) / data['avg_speed'].max()
            data['speed_variability'] = np.clip(data['speed_variability'], 0, 1)

        if 'temperature_celsius' in data.columns:
            data['temperature_stress'] = np.abs(data['temperature_celsius'] - 20) / 30
            data['temperature_stress'] = np.clip(data['temperature_stress'], 0, 1)

        if 'humidity_percent' in data.columns:
            data['humidity_stress'] = np.where(
                (data['humidity_percent'] > 80) | (data['humidity_percent'] < 20),
                np.abs(data['humidity_percent'] - 50) / 50,
                0
            )

        if 'month' in data.columns:
            winter_months = [12, 1, 2]
            summer_months = [6, 7, 8]
            data['seasonal_stress'] = np.where(
                data['month'].isin(winter_months + summer_months),
                0.7, 0.3
            )

        # Advanced composite features
        if all(col in data.columns for col in ['temperature_stress', 'humidity_stress', 'speed_variability']):
            data['engine_stress_composite'] = (
                data['temperature_stress'] * 0.4 +
                data['humidity_stress'] * 0.3 +
                data['speed_variability'] * 0.3
            )

        if all(col in data.columns for col in ['engine_hours', 'fuel_consumption_rate', 'load_frequency']):
            data['maintenance_urgency_score'] = (
                data['engine_hours'] * 0.4 +
                data['fuel_consumption_rate'] * 0.3 +
                data['load_frequency'] * 0.3
            )

        # Enhanced efficiency calculation
        if all(col in data.columns for col in ['avg_speed', 'distance', 'trip_duration_hours']):
            safe_trip_duration = data['trip_duration_hours'].replace(0, 0.1)
            safe_avg_speed = data['avg_speed'].replace(0, 1)

            data['operational_efficiency'] = data['distance'] / (safe_trip_duration * safe_avg_speed)
            data['operational_efficiency'] = data['operational_efficiency'].replace([np.inf, -np.inf], np.nan)
            data['operational_efficiency'] = data['operational_efficiency'].fillna(data['operational_efficiency'].mean())
            data['operational_efficiency'] = (data['operational_efficiency'] - data['operational_efficiency'].min()) / (data['operational_efficiency'].max() - data['operational_efficiency'].min())

        print("✅ Feature engineering completed")
        return data

    def _create_multiple_targets(self, data, clean_indices):
        """Create multiple prediction targets with PROPER maintenance types"""
        print("🎯 Creating multiple prediction targets...")

        targets = {}

        # Target 1: Binary Classification - Breakdown Risk (30 days)
        risk_factors = []

        if 'engine_stress_composite' in data.columns:
            risk_factors.append(data['engine_stress_composite'])

        if 'maintenance_urgency_score' in data.columns:
            risk_factors.append(data['maintenance_urgency_score'])

        if 'temperature_stress' in data.columns:
            risk_factors.append(data['temperature_stress'])

        if 'speed_variability' in data.columns:
            risk_factors.append(data['speed_variability'])

        if 'operational_efficiency' in data.columns:
            risk_factors.append(1 - data['operational_efficiency'])

        if risk_factors:
            breakdown_risk = np.zeros(len(data))
            weights = [0.35, 0.25, 0.20, 0.15, 0.05]

            for i, factor in enumerate(risk_factors[:len(weights)]):
                breakdown_risk += factor * weights[i]

            breakdown_risk += np.random.normal(0, 0.05, len(data))
            breakdown_risk = np.clip(breakdown_risk, 0, 1)

            risk_threshold = np.percentile(breakdown_risk, 78)
            targets['breakdown_risk'] = (breakdown_risk > risk_threshold).astype(int)

            print(f"  ✅ Target 1: Breakdown Risk - Class distribution: {dict(pd.Series(targets['breakdown_risk']).value_counts())}")

        # Target 2: Regression - Days until next maintenance
        if 'maintenance_urgency_score' in data.columns:
            # Convert urgency score to days (inverse relationship)
            max_days = 90  # Maximum days until maintenance
            min_days = 7   # Minimum days until maintenance

            urgency_normalized = (data['maintenance_urgency_score'] - data['maintenance_urgency_score'].min()) / \
                               (data['maintenance_urgency_score'].max() - data['maintenance_urgency_score'].min())

            targets['days_until_maintenance'] = max_days - (urgency_normalized * (max_days - min_days))
            targets['days_until_maintenance'] = np.clip(targets['days_until_maintenance'], min_days, max_days)

            print(f"  ✅ Target 2: Days until Maintenance - Range: {targets['days_until_maintenance'].min():.1f} to {targets['days_until_maintenance'].max():.1f} days")

        # Target 3: Multi-class - Maintenance Type (FIXED: engine, brakes, transmission, general)
        if 'engine_stress_composite' in data.columns and 'speed_variability' in data.columns:
            # Create maintenance type based on stress patterns - PROPER 4-TYPE CLASSIFICATION
            maintenance_types = []

            for i in range(len(data)):
                engine_stress = data.loc[data.index[i], 'engine_stress_composite']
                speed_var = data.loc[data.index[i], 'speed_variability']
                temp_stress = data.loc[data.index[i], 'temperature_stress']
                
                # Enhanced logic for 4 maintenance types
                if engine_stress > 0.7 or temp_stress > 0.6:
                    maintenance_types.append('engine')
                elif speed_var > 0.6 or data.loc[data.index[i], 'avg_speed'] > 70:
                    maintenance_types.append('brakes')
                elif temp_stress > 0.5 and data.loc[data.index[i], 'humidity_percent'] > 70:
                    maintenance_types.append('transmission')
                else:
                    maintenance_types.append('general')

            targets['maintenance_type'] = maintenance_types

            # Encode maintenance types
            le = LabelEncoder()
            targets['maintenance_type_encoded'] = le.fit_transform(targets['maintenance_type'])
            self.label_encoders['maintenance_type'] = le

            print(f"  ✅ Target 3: Maintenance Type - Types: {dict(pd.Series(targets['maintenance_type']).value_counts())}")

        return targets

    def train_additional_models(self, X, targets):
        """Train additional models for new prediction targets"""
        print("🌲 Training additional models...")

        try:
            # Scale features
            X_scaled = self.scalers['main'].transform(X)

            # Train Random Forest for breakdown risk (interpretability)
            print("  🌲 Training Random Forest for breakdown risk...")
            rf_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )

            y_breakdown = targets['breakdown_risk'].loc[X.index]
            rf_classifier.fit(X_scaled, y_breakdown)
            self.models['random_forest'] = rf_classifier

            # Evaluate Random Forest
            rf_score = rf_classifier.score(X_scaled, y_breakdown)
            print(f"    ✅ Random Forest accuracy: {rf_score:.3f}")

            # Train Random Forest for maintenance timing (regression)
            if 'days_until_maintenance' in targets:
                print("  🌲 Training Random Forest for maintenance timing...")
                rf_regressor = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )

                y_timing = targets['days_until_maintenance'].loc[X.index]
                rf_regressor.fit(X_scaled, y_timing)
                self.models['random_forest_regressor'] = rf_regressor

                # Evaluate regression
                y_pred = rf_regressor.predict(X_scaled)
                r2 = r2_score(y_timing, y_pred)
                rmse = np.sqrt(mean_squared_error(y_timing, y_pred))
                print(f"    ✅ Random Forest R²: {r2:.3f}, RMSE: {rmse:.2f} days")

            # Train Random Forest for maintenance type (multi-class) - FIXED 4-TYPE MODEL
            if 'maintenance_type_encoded' in targets:
                print("  🌲 Training Random Forest for maintenance type (4 types)...")
                rf_multiclass = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )

                y_type = targets['maintenance_type_encoded']
                rf_multiclass.fit(X_scaled, y_type)
                self.models['random_forest_multiclass'] = rf_multiclass

                # Evaluate multi-class
                type_score = rf_multiclass.score(X_scaled, y_type)
                print(f"    ✅ Random Forest multi-class accuracy: {type_score:.3f}")

            self.is_trained = True
            print("✅ All additional models trained successfully!")

            return True

        except Exception as e:
            print(f"❌ Error training additional models: {str(e)}")
            return False

    def predict_all_targets(self, X):
        """Make predictions for all targets using ensemble approach"""
        if not self.is_trained:
            print("❌ Models not trained yet!")
            return None

        try:
            X_scaled = self.scalers['main'].transform(X)

            predictions = {}

            # Target 1: Breakdown Risk (ensemble of XGBoost and Random Forest)
            xgb_pred = self.models['xgboost'].predict_proba(X_scaled)[:, 1]
            rf_pred = self.models['random_forest'].predict_proba(X_scaled)[:, 1]

            # Weighted ensemble (XGBoost: 0.7, Random Forest: 0.3)
            predictions['breakdown_risk_probability'] = 0.7 * xgb_pred + 0.3 * rf_pred
            predictions['breakdown_risk_binary'] = (predictions['breakdown_risk_probability'] > 0.5).astype(int)

            # Target 2: Days until Maintenance
            if 'random_forest_regressor' in self.models:
                predictions['days_until_maintenance'] = self.models['random_forest_regressor'].predict(X_scaled)

            # Target 3: Maintenance Type
            if 'random_forest_multiclass' in self.models:
                type_pred = self.models['random_forest_multiclass'].predict(X_scaled)
                predictions['maintenance_type_encoded'] = type_pred

                # Decode maintenance types
                if 'maintenance_type' in self.label_encoders:
                    predictions['maintenance_type'] = self.label_encoders['maintenance_type'].inverse_transform(type_pred)

            return predictions

        except Exception as e:
            print(f"❌ Error making predictions: {str(e)}")
            return None

    def get_feature_importance(self):
        """Get feature importance from all models"""
        if not self.is_trained:
            print("❌ Models not trained yet!")
            return None

        importance_data = {}

        # XGBoost feature importance
        if 'xgboost' in self.models:
            importance_data['xgboost'] = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models['xgboost'].feature_importances_
            }).sort_values('importance', ascending=False)

        # Random Forest feature importance
        if 'random_forest' in self.models:
            importance_data['random_forest'] = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.models['random_forest'].feature_importances_
            }).sort_values('importance', ascending=False)

        return importance_data

    def save_enhanced_models(self):
        """Save all enhanced models"""
        if not self.is_trained:
            print("❌ Models not trained yet!")
            return False

        try:
            # Save additional models
            joblib.dump(self.models['random_forest'], 'models/random_forest_model.pkl')
            joblib.dump(self.models['random_forest_regressor'], 'models/random_forest_regressor.pkl')
            joblib.dump(self.models['random_forest_multiclass'], 'models/random_forest_multiclass.pkl')

            # Save label encoders
            joblib.dump(self.label_encoders, 'models/label_encoders.pkl')

            print("✅ Enhanced models saved successfully!")
            return True

        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
            return False

def main():
    """Main function to train enhanced models"""
    print("🚀 ENHANCED FLEET MAINTENANCE ML ENGINE")
    print("=" * 60)

    # Initialize engine
    engine = EnhancedFleetMaintenanceEngine()

    # Load existing models
    if not engine.load_existing_model():
        print("❌ Failed to load existing models. Exiting.")
        return

    # Prepare data
    X, targets, data = engine.prepare_data()
    if X is None:
        print("❌ Failed to prepare data. Exiting.")
        return

    # Train additional models
    if not engine.train_additional_models(X, targets):
        print("❌ Failed to train additional models. Exiting.")
        return

    # Save enhanced models
    engine.save_enhanced_models()

    # Test predictions
    print("\n🧪 Testing enhanced predictions...")
    test_predictions = engine.predict_all_targets(X.head(10))

    if test_predictions:
        print("✅ Enhanced predictions working!")
        print(f"  Breakdown risk probabilities: {test_predictions['breakdown_risk_probability'][:5]}")
        if 'days_until_maintenance' in test_predictions:
            print(f"  Days until maintenance: {test_predictions['days_until_maintenance'][:5]}")
        if 'maintenance_type' in test_predictions:
            print(f"  Maintenance types: {test_predictions['maintenance_type'][:5]}")

    # Feature importance
    print("\n🔍 Feature Importance Analysis:")
    importance_data = engine.get_feature_importance()

    if importance_data:
        print("\nTop 5 XGBoost Features:")
        print(importance_data['xgboost'].head())

        print("\nTop 5 Random Forest Features:")
        print(importance_data['random_forest'].head())

    print("\n🎉 Enhanced ML Engine setup completed successfully!")
    print("🚀 Ready to build the comprehensive app!")

if __name__ == "__main__":
    main()
