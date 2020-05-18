from .CombinedClassifier import CombinedClassifier

mapping = {
    "Hemoglobin": "Hemoglobin",
    "Absolute Lymphocyte Count": "Absolute Lymphocyte Count",
    "Absolute Neut Count": "Absolute Neutrophil Count",
    "Absolute Baso Count": "Absolute Basophil Count",
    "Absolute Eos Count": "Absolute Eosinophil Count",
    "Absolute Mono Count": "Absolute Monocyte Count",
    "PLATELET COUNT, AUTO": "Platelet Count",
    "Ferritin": "Ferritin",
    "Lactate Dehydrogenase": "Lactate Dehydrogenase",
    "C-Reactive Protein": "C-Reactive Protein",
    "Red Blood Cell Count": "Red Blood Cell Count"
}

rev_mapping = {v: k for k, v in mapping.items()}