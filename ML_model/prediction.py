import torch 
from .MolNexTR import molnextr

def predict_from_image_file(image_file: str, 
                            model_path: str, 
                            device: str = None
                            ):
    """
    Predicts SMILES and molecular structure from a single image file path using a pre-trained MolNexTR model.
    
    Args:
        image_file (str): Path to the input image file.
        model_path (str): Path to the pre-trained model file.
        device (str): Device to run the model on ('cpu' or 'cuda').
        return_atoms_bonds (bool): Whether to return atom and bond information.
        
    Return:
        result (dict): Prediction result for the image file, with keys 'predicted_smiles', 'predicted_molfile', 'atom_sets', 'bond_sets'.

    """

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = molnextr(model_path=model_path, device=device)
    result = model.predict_final_results(image_file, return_atoms_bonds=True, return_confidence=True)
    
    return result