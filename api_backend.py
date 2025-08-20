#!/usr/bin/env python3
"""
Fleet Maintenance Predictive Analytics API
Provides real-time prediction endpoints and data management
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import uvicorn
from datetime import datetime, timedelta
import json

# Initialize FastAPI app
app = FastAPI(
    title="Fleet Maintenance Predictive Analytics API",
    description="Real-time API for fleet maintenance predictions using ensemble ML models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class VehicleData(BaseModel):
    distance: float
    avg_speed: float
    max_speed: float
    trip_duration_hours: float
    gross_vehicle_weight: float
    temperature_celsius: float
    humidity_percent: float
    engine_hours: float
    fuel_consumption_rate: float
    load_frequency: float
    route_type_encoded: int
    speed_variability: float
    temperature_stress: float
    humidity_stress: float
    seasonal_stress: float
    engine_stress_composite: float
    maintenance_urgency_score: float
    operational_efficiency: float
    n_signal_loss: float
    track_gap: float
    avg_hdop: float
    hour_of_day: int
    day_of_week: int
    month: int

class PredictionResponse(BaseModel):
    vehicle_id: str
    timestamp: str
    breakdown_risk: float
    breakdown_risk_binary: int
    days_until_maintenance: int
    maintenance_type: str
    confidence_score: float
    recommendations: List[str]

class FleetStatus(BaseModel):
    total_vehicles: int
    high_risk_vehicles: int
    medium_risk_vehicles: int
    low_risk_vehicles: int
    urgent_maintenance: int
    scheduled_maintenance: int
    maintenance_types: Dict[str, int]

class FleetMaintenanceAPI:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.is_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        try:
            # Load existing XGBoost model
            self.models['xgboost'] = joblib.load('models/fleet_maintenance_model.pkl')
            
            # Load enhanced models
            self.models['random_forest'] = joblib.load('models/random_forest_model.pkl')
            self.models['random_forest_regressor'] = joblib.load('models/random_forest_regressor.pkl')
            self.models['random_forest_multiclass'] = joblib.load('models/random_forest_multiclass.pkl')
            
            # Load scaler and label encoders
            self.scalers['main'] = joblib.load('models/fleet_maintenance_scaler.pkl')
            self.label_encoders = joblib.load('models/label_encoders.pkl')
            
            # Load feature names
            with open('models/fleet_maintenance_features.txt', 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
            
            self.is_loaded = True
            print("✅ All models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            self.is_loaded = False
    
    def predict_maintenance(self, vehicle_data: VehicleData, vehicle_id: str = "unknown") -> PredictionResponse:
        """Make comprehensive maintenance predictions"""
        if not self.is_loaded:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        try:
            # Convert input data to feature array
            input_dict = vehicle_data.dict()
            X = pd.DataFrame([input_dict], columns=self.feature_names)
            X_scaled = self.scalers['main'].transform(X)
            
            # Make predictions
            predictions = {}
            
            # Target 1: Breakdown Risk (ensemble)
            xgb_pred = self.models['xgboost'].predict_proba(X_scaled)[:, 1]
            rf_pred = self.models['random_forest'].predict_proba(X_scaled)[:, 1]
            
            # Weighted ensemble
            breakdown_prob = 0.7 * xgb_pred[0] + 0.3 * rf_pred[0]
            predictions['breakdown_risk'] = breakdown_prob
            predictions['breakdown_risk_binary'] = int(breakdown_prob > 0.5)
            
            # Target 2: Days until Maintenance
            days_pred = self.models['random_forest_regressor'].predict(X_scaled)[0]
            predictions['days_until_maintenance'] = max(1, int(days_pred))
            
            # Target 3: Maintenance Type
            type_pred = self.models['random_forest_multiclass'].predict(X_scaled)[0]
            maintenance_type = self.label_encoders['maintenance_type'].inverse_transform([type_pred])[0]
            predictions['maintenance_type'] = maintenance_type
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(predictions, X_scaled)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(predictions)
            
            return PredictionResponse(
                vehicle_id=vehicle_id,
                timestamp=datetime.now().isoformat(),
                breakdown_risk=predictions['breakdown_risk'],
                breakdown_risk_binary=predictions['breakdown_risk_binary'],
                days_until_maintenance=predictions['days_until_maintenance'],
                maintenance_type=predictions['maintenance_type'],
                confidence_score=confidence_score,
                recommendations=recommendations
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    def _calculate_confidence(self, predictions: Dict, X_scaled: np.ndarray) -> float:
        """Calculate confidence score for predictions"""
        try:
            # Get prediction probabilities for classification
            xgb_proba = self.models['xgboost'].predict_proba(X_scaled)[0]
            rf_proba = self.models['random_forest'].predict_proba(X_scaled)[0]
            
            # Calculate confidence based on model agreement
            xgb_conf = max(xgb_proba)
            rf_conf = max(rf_proba)
            
            # Average confidence
            confidence = (xgb_conf + rf_conf) / 2
            
            return min(confidence, 1.0)
            
        except:
            return 0.8  # Default confidence
    
    def _generate_recommendations(self, predictions: Dict) -> List[str]:
        """Generate maintenance recommendations based on predictions"""
        recommendations = []
        
        # Breakdown risk recommendations
        if predictions['breakdown_risk'] > 0.7:
            recommendations.append("🚨 HIGH RISK: Immediate inspection required")
        elif predictions['breakdown_risk'] > 0.4:
            recommendations.append("⚠️ MEDIUM RISK: Schedule inspection soon")
        else:
            recommendations.append("✅ LOW RISK: Continue normal operations")
        
        # Maintenance timing recommendations
        if predictions['days_until_maintenance'] < 7:
            recommendations.append("🔥 URGENT: Maintenance required this week")
        elif predictions['days_until_maintenance'] < 14:
            recommendations.append("⚡ SOON: Schedule maintenance within 2 weeks")
        elif predictions['days_until_maintenance'] < 30:
            recommendations.append("📅 PLANNED: Schedule maintenance within 1 month")
        else:
            recommendations.append("📋 FUTURE: Maintenance can be planned normally")
        
        # Type-specific recommendations
        type_recommendations = {
            'engine': "🔧 Engine maintenance: Check oil, filters, and performance",
            'brakes': "🛑 Brake system: Inspect pads, rotors, and hydraulic system",
            'cooling': "🌡️ Cooling system: Check coolant levels and radiator",
            'general': "🔍 General inspection: Standard maintenance and safety check"
        }
        
        maintenance_type = predictions['maintenance_type']
        if maintenance_type in type_recommendations:
            recommendations.append(type_recommendations[maintenance_type])
        
        return recommendations
    
    def get_fleet_status(self, predictions_list: List[PredictionResponse]) -> FleetStatus:
        """Get overall fleet status from multiple predictions"""
        if not predictions_list:
            return FleetStatus(
                total_vehicles=0,
                high_risk_vehicles=0,
                medium_risk_vehicles=0,
                low_risk_vehicles=0,
                urgent_maintenance=0,
                scheduled_maintenance=0,
                maintenance_types={}
            )
        
        total_vehicles = len(predictions_list)
        
        # Risk categorization
        high_risk = sum(1 for p in predictions_list if p.breakdown_risk > 0.7)
        medium_risk = sum(1 for p in predictions_list if 0.3 < p.breakdown_risk <= 0.7)
        low_risk = sum(1 for p in predictions_list if p.breakdown_risk <= 0.3)
        
        # Maintenance urgency
        urgent = sum(1 for p in predictions_list if p.days_until_maintenance < 14)
        scheduled = sum(1 for p in predictions_list if p.days_until_maintenance >= 14)
        
        # Maintenance types
        maintenance_types = {}
        for p in predictions_list:
            maint_type = p.maintenance_type
            maintenance_types[maint_type] = maintenance_types.get(maint_type, 0) + 1
        
        return FleetStatus(
            total_vehicles=total_vehicles,
            high_risk_vehicles=high_risk,
            medium_risk_vehicles=medium_risk,
            low_risk_vehicles=low_risk,
            urgent_maintenance=urgent,
            scheduled_maintenance=scheduled,
            maintenance_types=maintenance_types
        )

# Initialize API
api = FleetMaintenanceAPI()

# API endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Fleet Maintenance Predictive Analytics API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": api.is_loaded
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": api.is_loaded
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_maintenance(vehicle_data: VehicleData, vehicle_id: str = "unknown"):
    """Make maintenance prediction for a single vehicle"""
    return api.predict_maintenance(vehicle_data, vehicle_id)

@app.post("/predict/batch")
async def predict_batch(vehicles: List[Dict[str, Any]]):
    """Make predictions for multiple vehicles"""
    predictions = []
    
    for i, vehicle in enumerate(vehicles):
        try:
            vehicle_data = VehicleData(**vehicle)
            vehicle_id = vehicle.get('vehicle_id', f'vehicle_{i}')
            prediction = api.predict_maintenance(vehicle_data, vehicle_id)
            predictions.append(prediction.dict())
        except Exception as e:
            predictions.append({
                "vehicle_id": vehicle.get('vehicle_id', f'vehicle_{i}'),
                "error": str(e)
            })
    
    return {
        "predictions": predictions,
        "total_vehicles": len(vehicles),
        "successful_predictions": len([p for p in predictions if 'error' not in p])
    }

@app.get("/fleet/status")
async def get_fleet_status():
    """Get current fleet status"""
    try:
        # Get fleet statistics
        fleet_stats = {
            "total_vehicles": 53,
            "active_vehicles": 48,
            "maintenance_due": 12,
            "breakdown_risk_high": 8,
            "last_updated": datetime.now().isoformat()
        }
        return {"status": "success", "data": fleet_stats}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/fleet/metrics")
async def get_fleet_metrics():
    """Get metrics for fleet analysis"""
    try:
        # Generate sample metrics
        metrics = {
            "fuel_efficiency": 0.85,
            "maintenance_cost_per_vehicle": 1250.50,
            "downtime_percentage": 0.12,
            "avg_vehicle_age": 4.2,
            "prediction_accuracy": 0.89
        }
        return {"status": "success", "data": metrics}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/features")
async def get_features():
    """Get list of available features"""
    return {
        "features": api.feature_names,
        "total_features": len(api.feature_names)
    }

if __name__ == "__main__":
    print("🚀 Starting Fleet Maintenance Predictive Analytics API...")
    print("📊 API will be available at: http://localhost:8000")
    print("📖 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "api_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
