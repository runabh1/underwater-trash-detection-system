# Underwater Trash Detection Classes
# This file contains the class names for the actual 15 classes the model was trained on

# Actual classes from the trained model
TRASH_CLASSES = {
    0: "mask",
    1: "can", 
    2: "cellphone",
    3: "electronics",
    4: "gbottle",
    5: "glove",
    6: "metal",
    7: "misc",
    8: "net",
    9: "pbag",
    10: "pbottle",
    11: "plastic",
    12: "rod",
    13: "sunglasses",
    14: "tyre"
}

# Display names for better readability
TRASH_CLASSES_SHORT = {
    0: "Mask",
    1: "Can", 
    2: "Cellphone",
    3: "Electronics",
    4: "Glass Bottle",
    5: "Glove",
    6: "Metal",
    7: "Misc",
    8: "Net",
    9: "Plastic Bag",
    10: "Plastic Bottle",
    11: "Plastic",
    12: "Rod",
    13: "Sunglasses",
    14: "Tyre"
}

# Color mapping for different trash types (BGR format for OpenCV)
TRASH_COLORS = {
    0: (0, 255, 0),      # Green for mask
    1: (255, 0, 0),      # Blue for can
    2: (255, 255, 0),    # Cyan for cellphone
    3: (0, 0, 255),      # Red for electronics
    4: (255, 255, 255),  # White for glass bottle
    5: (0, 255, 255),    # Yellow for glove
    6: (128, 128, 128),  # Gray for metal
    7: (255, 0, 255),    # Magenta for misc
    8: (0, 128, 255),    # Orange for net
    9: (0, 255, 128),    # Light green for plastic bag
    10: (128, 0, 128),   # Purple for plastic bottle
    11: (255, 128, 0),   # Light blue for plastic
    12: (128, 0, 0),     # Dark blue for rod
    13: (255, 255, 128), # Light yellow for sunglasses
    14: (0, 128, 0)      # Dark green for tyre
}

def get_class_name(class_id):
    """Get the class name for a given class ID"""
    return TRASH_CLASSES.get(class_id, f"Unknown_{class_id}")

def get_class_name_short(class_id):
    """Get the short class name for a given class ID"""
    return TRASH_CLASSES_SHORT.get(class_id, f"Unknown_{class_id}")

def get_class_color(class_id):
    """Get the color for a given class ID"""
    return TRASH_COLORS.get(class_id, (0, 255, 0))  # Default green

def get_all_classes():
    """Get all class names"""
    return TRASH_CLASSES

def get_all_classes_short():
    """Get all short class names"""
    return TRASH_CLASSES_SHORT 