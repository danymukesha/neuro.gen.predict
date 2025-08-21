#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pickle
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

class NeuroGenPredict:
    """
    Alzheimer's Disease Genetic Risk Prediction Tool
    
    This class implements a novel ensemble approach that combines multiple
    genetic risk assessment methods optimized for computational efficiency
    and interpretability.
    """
    
    def __init__(self, population: str = "EUR"):
        """
        Initialize the predictor with population-specific parameters
        
        Args:
            population: Population ancestry code (EUR, AFR, EAS, etc.)
        """
        self.population = population
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.pathway_weights = {}
        
        # Known AD risk variants from public GWAS (no access restrictions needed)
        # These are well-established variants from published literature
        self.ad_variants = {
            'APOE_e4': {'weight': 3.68, 'chr': 19, 'pos': 45411941},  # rs429358
            'APOE_e2': {'weight': 0.6, 'chr': 19, 'pos': 45412079},   # rs7412
            'BIN1': {'weight': 1.15, 'chr': 2, 'pos': 127892810},     # rs744373
            'CLU': {'weight': 1.16, 'chr': 8, 'pos': 27467686},       # rs11136000
            'ABCA7': {'weight': 1.23, 'chr': 19, 'pos': 1063443},     # rs3764650
            'CR1': {'weight': 1.18, 'chr': 1, 'pos': 207692049},      # rs6656401
            'PICALM': {'weight': 1.13, 'chr': 11, 'pos': 85867875},   # rs3851179
            'MS4A6A': {'weight': 1.12, 'chr': 11, 'pos': 60009906},   # rs610932
            'CD33': {'weight': 1.10, 'chr': 19, 'pos': 51728477},     # rs3865444
            'MS4A4E': {'weight': 1.09, 'chr': 11, 'pos': 59923508},   # rs670139
            'CD2AP': {'weight': 1.10, 'chr': 6, 'pos': 47487762},     # rs9349407
            'EPHA1': {'weight': 1.11, 'chr': 7, 'pos': 143110762},    # rs11767557
            'TREM2': {'weight': 2.92, 'chr': 6, 'pos': 41129252},     # rs75932628
            'SORL1': {'weight': 1.14, 'chr': 11, 'pos': 121435587}    # rs2070045
        }
        
        # Biological pathway definitions for pathway-based analysis
        self.pathways = {
            'amyloid_processing': ['APOE', 'PSEN1', 'PSEN2', 'APP', 'BACE1', 'ADAM10'],
            'tau_pathology': ['MAPT', 'STH', 'KANSL1'],
            'inflammation': ['TREM2', 'CD33', 'INPP5D', 'MEF2C', 'MS4A6A'],
            'lipid_metabolism': ['APOE', 'CLU', 'ABCA7', 'SORL1'],
            'synaptic_function': ['PICALM', 'BIN1', 'CD2AP', 'EPHA1'],
            'immune_response': ['CR1', 'MS4A4E', 'HLA-DRB1', 'PLCG2']
        }
    
    def calculate_prs(self, genotype_data: pd.DataFrame) -> np.ndarray:
        """
        Calculate Polygenic Risk Score using established AD variants
        
        Args:
            genotype_data: DataFrame with variant genotypes (0, 1, 2 format)
            
        Returns:
            Array of PRS values for each sample
        """
        prs_scores = np.zeros(len(genotype_data))
        
        for variant, info in self.ad_variants.items():
            if variant in genotype_data.columns:
                # Apply additive genetic model with log-odds weighting
                variant_contribution = genotype_data[variant] * np.log(info['weight'])
                prs_scores += variant_contribution
            else:
                print(f"Warning: {variant} not found in genotype data")
        
        return prs_scores
    
    def calculate_pathway_scores(self, genotype_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pathway-specific genetic burden scores
        
        This novel approach aggregates genetic variants within biological
        pathways to provide interpretable risk components
        """
        pathway_scores = pd.DataFrame()
        
        for pathway, genes in self.pathways.items():
            pathway_variants = []
            for gene in genes:
                # Find variants associated with this gene
                gene_variants = [col for col in genotype_data.columns 
                                 if gene.upper() in col.upper()]
                pathway_variants.extend(gene_variants)
            
            if pathway_variants:
                # Calculate burden score as weighted sum of variants in pathway
                pathway_score = genotype_data[pathway_variants].sum(axis=1)
                pathway_scores[f'{pathway}_burden'] = pathway_score
            else:
                pathway_scores[f'{pathway}_burden'] = 0
        
        return pathway_scores
    
    def population_adjustment(self, scores: np.ndarray) -> np.ndarray:
        """
        Apply population-specific risk adjustments
        
        This addresses the limitation of many tools not accounting for
        population stratification in risk assessment
        """
        # Population-specific adjustment factors based on known AD prevalence
        adjustment_factors = {
            'EUR': 1.0,      # European (baseline)
            'AFR': 0.8,      # African ancestry (lower APOE e4 effect)
            'EAS': 0.9,      # East Asian
            'AMR': 0.95,     # Admixed American
            'SAS': 1.05      # South Asian
        }
        
        factor = adjustment_factors.get(self.population, 1.0)
        return scores * factor
    
    def prepare_features(self, genotype_data: pd.DataFrame, 
                         clinical_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepare comprehensive feature matrix for prediction
        
        Args:
            genotype_data: Genetic variant data
            clinical_data: Optional clinical/demographic data
            
        Returns:
            Feature matrix ready for machine learning
        """
        features = pd.DataFrame()
        
        # 1. Polygenic Risk Score
        prs = self.calculate_prs(genotype_data)
        features['PRS'] = prs
        
        # 2. Pathway-specific burden scores
        pathway_scores = self.calculate_pathway_scores(genotype_data)
        features = pd.concat([features, pathway_scores], axis=1)
        
        # 3. Population-adjusted scores
        features['PRS_adjusted'] = self.population_adjustment(prs)
        
        # 4. Individual high-impact variants (for interpretability)
        high_impact_variants = ['APOE_e4', 'TREM2', 'APOE_e2']
        for variant in high_impact_variants:
            if variant in genotype_data.columns:
                features[variant] = genotype_data[variant]
        
        # 5. Genetic interaction terms (novel feature)
        if all(v in genotype_data.columns for v in ['APOE_e4', 'TREM2']):
            features['APOE_TREM2_interaction'] = (genotype_data['APOE_e4'] * 
                                                  genotype_data['TREM2'])
        
        # 6. Clinical data integration (if available)
        if clinical_data is not None:
            clinical_features = ['age', 'sex', 'education_years']
            for feature in clinical_features:
                if feature in clinical_data.columns:
                    features[feature] = clinical_data[feature]
        
        self.feature_names = features.columns.tolist()
        return features
    
    def train_ensemble(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        Train ensemble of optimized models for AD risk prediction
        
        Uses computationally efficient algorithms optimized for small datasets
        """
        # Scale features for better convergence
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize ensemble components
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=100,           # Reduced for computational efficiency
                max_depth=10,               # Prevent overfitting
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced',    # Handle class imbalance
                n_jobs=-1                   # Use all available cores
            ),
            'gbm': GradientBoostingClassifier(
                n_estimators=100,           # Efficient ensemble size
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                subsample=0.8,              # Stochastic boosting for robustness
                random_state=42
            )
        }
        
        # Train models and evaluate performance
        cv_scores = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            model.fit(X_scaled, y)
            scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
            cv_scores[f'{name}_auc'] = scores.mean()
            cv_scores[f'{name}_std'] = scores.std()
            
            print(f"{name.upper()} Cross-validation AUC: {scores.mean():.3f} ± {scores.std():.3f}")
        
        return cv_scores
    
    def predict_risk(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate comprehensive risk predictions with interpretability
        
        Returns:
            Dictionary containing risk scores, probabilities, and feature importance
        """
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        
        # Ensemble predictions
        rf_proba = self.models['rf'].predict_proba(X_scaled)[:, 1]
        gbm_proba = self.models['gbm'].predict_proba(X_scaled)[:, 1]
        
        # Weighted ensemble (RF gets higher weight due to interpretability)
        ensemble_proba = 0.6 * rf_proba + 0.4 * gbm_proba
        
        predictions['risk_probability'] = ensemble_proba
        predictions['risk_category'] = self._categorize_risk(ensemble_proba)
        
        # Feature importance for interpretability
        rf_importance = self.models['rf'].feature_importances_
        predictions['feature_importance'] = dict(zip(self.feature_names, rf_importance))
        
        return predictions
    
    def _categorize_risk(self, probabilities: np.ndarray) -> List[str]:
        """
        Categorize continuous risk probabilities into interpretable categories
        """
        categories = []
        for prob in probabilities:
            if prob < 0.2:
                categories.append('Low Risk')
            elif prob < 0.5:
                categories.append('Moderate Risk')
            elif prob < 0.8:
                categories.append('High Risk')
            else:
                categories.append('Very High Risk')
        return categories
    
    def generate_report(self, predictions: Dict[str, np.ndarray], 
                        sample_id: str = "Sample", index: int = 0) -> str:
        """
        Generate interpretable clinical report
        
        This addresses the key gap in explainability that current tools lack
        """
        risk_prob = predictions['risk_probability'][index]
        risk_cat = predictions['risk_category'][index]
        
        # Sort feature importance
        importance = predictions['feature_importance']
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        report = f"""
        ===== NeuroGenPredict Risk Assessment Report =====
        
        Sample ID: {sample_id}
        Risk Probability: {risk_prob:.1%}
        Risk Category: {risk_cat}
        Population: {self.population}
        
        Top Contributing Factors:
        """
        
        for feature, importance_score in sorted_features:
            report += f"  • {feature}: {importance_score:.3f}\n"
        
        report += f"""
        
        Clinical Interpretation:
        - This assessment is based on established genetic variants
        - Risk is calculated using population-adjusted algorithms
        - Results should be interpreted alongside clinical evaluation
        - Genetic risk represents predisposition, not certainty
        
        Recommendations:
        - {self._generate_recommendations(risk_prob)}
        """
        
        return report
    
    def _generate_recommendations(self, risk_prob: float) -> str:
        """Generate risk-appropriate clinical recommendations"""
        if risk_prob < 0.2:
            return "Continue routine health monitoring and lifestyle maintenance"
        elif risk_prob < 0.5:
            return "Consider lifestyle modifications and regular cognitive assessment"
        elif risk_prob < 0.8:
            return "Recommend genetic counseling and enhanced monitoring"
        else:
            return "Urgent genetic counseling and comprehensive neurological evaluation recommended"
    
    def save(self, path: str):
        """
        Save the trained model to a file.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        """
        Load a trained model from a file.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
