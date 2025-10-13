import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors

class Predictor:
    def __init__(self, property_list=None):
        """
        Loads the trained models from disk. 
        """
        # If no list is provided, assume the four properties
        if property_list is None:
            property_list = ["S. aureus","E. faecalis","E. coli","P. aeruginosa"]
        
        self.property_list = property_list
        self.models = {}
        
        # Load each model from its .pkl file
        for prop_col in self.property_list:
            model_filename = f"random_forest_models/{prop_col}.pkl"
            self.models[prop_col] = joblib.load(model_filename)

        # Load the prefence classification model
        self.preference_model = joblib.load("random_forest_models/preference.pkl")
    
    def compute_rdkit_descriptors(self, smiles):
        """
        Compute RDKit descriptors for a single SMILES.
        Return a 1D numpy array of descriptors.
        If SMILES is invalid, return None.
        """
        mol = Chem.MolFromSmiles(smiles)
        descriptor_names_and_functions = Descriptors._descList
        if mol is None:
            return None
        
        descriptor_values = []
        for _, func in descriptor_names_and_functions:
            descriptor_values.append(func(mol))
        return np.array(descriptor_values).reshape(1, -1)  # shape (1, n_descriptors)
    
    def predict_properties(self, smiles):
        """
        Compute the descriptors for the given SMILES and then predict 
        each property using the loaded models.
        
        Return a dictionary of {property_name: predicted_value}.
        If SMILES is invalid, return None or an empty dict.
        """
        if smiles == "" or smiles == None:
            return None
        descriptors = self.compute_rdkit_descriptors(smiles)
        if descriptors is None:
            # Invalid SMILES
            return None
        
        results = {}
        for prop_col in self.property_list:
            model = self.models[prop_col]
            pred = model.predict(descriptors)  # descriptors has shape (1, n_features)
            results[prop_col] = float(pred[0])  # Convert to Python float
        return results

    def predict_preference(self, smiles):
        """
        Predict the preference for the given SMILES using the loaded preference model.
        Return the predicted preference as a string.
        If SMILES is invalid, return None.
        """
        if smiles == "" or smiles == None:
            return None
        descriptors = self.compute_rdkit_descriptors(smiles)
        if descriptors is None:
            # Invalid SMILES
            return None
        
        pred = self.preference_model.predict(descriptors)
        return pred[0]

# Example usage (if you run predictor.py directly):
if __name__ == "__main__":
    p = Predictor()
    test_smiles = "CCO"  # Ethanol
    props = p.predict_properties(test_smiles)
    print(f"Predictions for SMILES='{test_smiles}': {props}")
