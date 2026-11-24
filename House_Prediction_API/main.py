
import io
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field


app = FastAPI(
    title="California House Price Prediction API",
    description="Predict house prices using Random Forest Regressor",
    version="1.0"
)


# Load trained ML model and feature names
model = joblib.load("house_model.joblib")
features = joblib.load("house_features.joblib")


# Input schema
class HouseFeatures(BaseModel):

    MedInc: float = Field(gt=0, description="Median income of neighborhood")
    HouseAge: float = Field(ge=0, description="Average age of house")
    AveRooms: float = Field(gt=0, description="Average number of rooms")
    AveBedrms: float = Field(gt=0, description="Average number of bedrooms")
    Population: float = Field(gt=0, description="Total population")
    AveOccup: float = Field(gt=0, description="Average number of occupants")
    Latitude: float = Field(ge=32, le=42, description="Latitude")
    Longitude: float = Field(ge=-125, le=-114, description="Longitude")


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "California House Prediction API",
        "status": "running",
        "endpoint": "Send POST request to /predict"
    }


# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "RandomForestRegressor",
        "features": features,
        "average_error": "$39,000"
    }


# Single house price prediction endpoint
@app.post("/predict")
def predict(house: HouseFeatures):

    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])

        # Arrange columns in training order
        input_data = input_data[features]

        # Predict house price
        predicted = model.predict(input_data)[0]

        # Convert prediction into USD
        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.2f}",
            "predicted_price_in_hundred_thousands": f"${predicted:.4f}",
            "estimated_range": (
                f"${price_usd - 39000:,.0f} "
                f"to ${price_usd + 39000:,.0f}"
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# Multiple house price predictions using CSV file
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):

    try:
        # Check file extension
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Please upload a CSV file only"
            )

        # Read uploaded file
        contents = await file.read()

        # Convert CSV data into DataFrame
        df = pd.read_csv(io.BytesIO(contents))

        # Check whether CSV is empty
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV file is empty"
            )

        # Required columns
        required_columns = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude"
        ]

        # Find missing columns
        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        # Return error if columns are missing
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns in CSV file: {missing_columns}"
            )

        # Select required columns in training order
        input_data = df[features].copy()

        # Convert columns to numeric values
        input_data = input_data.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Check for missing or invalid values
        if input_data.isnull().any().any():
            raise HTTPException(
                status_code=400,
                detail="CSV contains missing or invalid numeric values"
            )

        # Generate predictions for all rows
        predictions = model.predict(input_data)

        # Convert predictions into USD
        prices_usd = predictions * 100000

        # Add predictions to original DataFrame
        df["Predicted_Price_USD"] = [
            round(float(price), 2) for price in prices_usd
        ]

        # Return results
        return {
            "message": "Predictions generated successfully",
            "total_records": len(df),
            "predictions": df.to_dict(orient="records")
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File prediction failed: {str(e)}"
        )

    finally:
        await file.close()
