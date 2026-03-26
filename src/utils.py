import os

def get_versioned_path(base_path):
    """
    Increments file versions to prevent overwriting.
    Example: submission.csv -> submission_v2.csv
    """
    if not os.path.exists(base_path):
        return base_path
    
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    
    counter = 2
    new_path = os.path.join(directory, f"{name}_v{counter}{ext}")
    
    while os.path.exists(new_path):
        counter += 1
        new_path = os.path.join(directory, f"{name}_v{counter}{ext}")
        
    return new_path
