import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Data processing utilities for health analytics"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        
    def process_data(self, df):
        """Process raw health data for analysis"""
        try:
            # Make a copy to avoid modifying original data
            processed_df = df.copy()
            
            # Handle date columns
            processed_df = self._process_dates(processed_df)
            
            # Handle missing values
            processed_df = self._handle_missing_values(processed_df)
            
            # Validate and clean data
            processed_df = self._validate_data(processed_df)
            
            # Create derived features
            processed_df = self._create_derived_features(processed_df)
            
            # Normalize numerical features
            processed_df = self._normalize_features(processed_df)
            
            return processed_df
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return None
    
    def _process_dates(self, df):
        """Process date columns"""
        # Common date column names
        date_cols = ['date', 'timestamp', 'created_at', 'recorded_at']
        
        for col in date_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    
                    # Extract date features
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_weekday'] = df[col].dt.weekday
                    df[f'{col}_hour'] = df[col].dt.hour
                    
                except Exception as e:
                    print(f"Error processing date column {col}: {str(e)}")
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        # Handle numerical missing values
        if numerical_cols:
            imputer_num = SimpleImputer(strategy='median')
            df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])
            self.imputers['numerical'] = imputer_num
        
        # Handle categorical missing values
        if categorical_cols:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
            self.imputers['categorical'] = imputer_cat
        
        return df
    
    def _validate_data(self, df):
        """Validate and clean health data"""
        # Health-specific validation rules
        validation_rules = {
            'sleep_hours': (0, 24),
            'exercise_minutes': (0, 480),  # 8 hours max
            'mood_score': (1, 10),
            'stress_level': (1, 10),
            'heart_rate': (30, 200),
            'steps': (0, 50000),
            'calories_burned': (0, 5000),
            'water_intake': (0, 5000),  # ml
            'weight': (30, 300),  # kg
            'body_temperature': (35, 42),  # Celsius
            'blood_pressure_systolic': (70, 200),
            'blood_pressure_diastolic': (40, 120)
        }
        
        for col, (min_val, max_val) in validation_rules.items():
            if col in df.columns:
                # Cap values at reasonable limits
                df[col] = df[col].clip(lower=min_val, upper=max_val)
        
        return df
    
    def _create_derived_features(self, df):
        """Create derived features from existing data"""
        # BMI calculation
        if 'weight' in df.columns and 'height' in df.columns:
            # Convert height from cm to m
            height_m = df['height'] / 100
            df['bmi'] = df['weight'] / (height_m ** 2)
        
        # Sleep efficiency
        if 'sleep_hours' in df.columns and 'time_in_bed' in df.columns:
            df['sleep_efficiency'] = (df['sleep_hours'] / df['time_in_bed']) * 100
        
        # Activity intensity
        if 'exercise_minutes' in df.columns and 'calories_burned' in df.columns:
            df['exercise_intensity'] = df['calories_burned'] / df['exercise_minutes'].replace(0, 1)
        
        # Stress-mood ratio
        if 'stress_level' in df.columns and 'mood_score' in df.columns:
            df['stress_mood_ratio'] = df['stress_level'] / df['mood_score'].replace(0, 1)
        
        # Daily activity score
        if 'steps' in df.columns and 'exercise_minutes' in df.columns:
            df['daily_activity_score'] = (df['steps'] / 1000) + (df['exercise_minutes'] / 10)
        
        # Hydration per weight
        if 'water_intake' in df.columns and 'weight' in df.columns:
            df['hydration_per_kg'] = df['water_intake'] / df['weight']
        
        # Sleep debt (assuming 8 hours is optimal)
        if 'sleep_hours' in df.columns:
            df['sleep_debt'] = 8 - df['sleep_hours']
            df['sleep_debt'] = df['sleep_debt'].clip(lower=0)
        
        return df
    
    def _normalize_features(self, df):
        """Normalize numerical features"""
        # Get numerical columns (excluding derived date features)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude ID columns and other non-scalable features
        exclude_cols = [col for col in numerical_cols if any(x in col.lower() for x in ['id', 'year', 'month', 'day', 'hour', 'weekday'])]
        scalable_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if scalable_cols:
            # Store original values for reference
            for col in scalable_cols:
                df[f'{col}_original'] = df[col]
            
            # Apply scaling
            df[scalable_cols] = self.scaler.fit_transform(df[scalable_cols])
        
        return df
    
    def inverse_transform(self, df, columns):
        """Inverse transform normalized features"""
        try:
            df_copy = df.copy()
            df_copy[columns] = self.scaler.inverse_transform(df_copy[columns])
            return df_copy
        except Exception as e:
            print(f"Error in inverse transform: {str(e)}")
            return df
    
    def encode_categorical_features(self, df):
        """Encode categorical features"""
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            
            df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        return df
    
    def create_time_series_features(self, df, date_col='date'):
        """Create time series specific features"""
        if date_col not in df.columns:
            return df
        
        # Sort by date
        df = df.sort_values(date_col)
        
        # Create lag features for key metrics
        lag_features = ['sleep_hours', 'exercise_minutes', 'mood_score', 'stress_level']
        
        for feature in lag_features:
            if feature in df.columns:
                # 1-day lag
                df[f'{feature}_lag1'] = df[feature].shift(1)
                
                # 7-day lag
                df[f'{feature}_lag7'] = df[feature].shift(7)
                
                # Rolling averages
                df[f'{feature}_rolling_7d'] = df[feature].rolling(window=7, min_periods=1).mean()
                df[f'{feature}_rolling_30d'] = df[feature].rolling(window=30, min_periods=1).mean()
                
                # Trend (difference from previous day)
                df[f'{feature}_trend'] = df[feature] - df[feature].shift(1)
        
        return df
    
    def detect_outliers(self, df, method='iqr'):
        """Detect outliers in the dataset"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outliers = {}
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers[col] = df[z_scores > 3].index.tolist()
        
        return outliers
    
    def clean_outliers(self, df, method='cap'):
        """Clean outliers from the dataset"""
        outliers = self.detect_outliers(df)
        
        for col, outlier_indices in outliers.items():
            if len(outlier_indices) > 0:
                if method == 'cap':
                    # Cap at 95th and 5th percentiles
                    lower_cap = df[col].quantile(0.05)
                    upper_cap = df[col].quantile(0.95)
                    df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
                
                elif method == 'remove':
                    df = df.drop(outlier_indices)
        
        return df
    
    def get_data_summary(self, df):
        """Get comprehensive data summary"""
        summary = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numerical_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
            'categorical_summary': {col: df[col].value_counts().to_dict() for col in df.select_dtypes(include=['object']).columns}
        }
        
        return summary
