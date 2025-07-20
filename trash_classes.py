# Underwater Trash Detection Classes
# This file contains the class names for the actual 15 classes the model was trained on

# FLEXIBLE MAPPING: This will be updated based on actual model behavior
# The issue is that class IDs from the model don't match our assumption

# Expected class names in the order you provided
EXPECTED_CLASSES = ["mask", "can", "cellphone", "electronics", "gbottle", "glove", "metal", "misc", "net", "pbag", "pbottle", "plastic", "rod", "sunglasses", "tyre"]

# Display names for better readability
DISPLAY_NAMES = ["Mask", "Can", "Cellphone", "Electronics", "Glass Bottle", "Glove", "Metal", "Misc", "Net", "Plastic Bag", "Plastic Bottle", "Plastic", "Rod", "Sunglasses", "Tyre"]

# Colors for different trash types (BGR format for OpenCV)
COLORS = [
    (0, 255, 0),      # Green for mask
    (255, 0, 0),      # Blue for can
    (255, 255, 0),    # Cyan for cellphone
    (0, 0, 255),      # Red for electronics
    (255, 255, 255),  # White for glass bottle
    (0, 255, 255),    # Yellow for glove
    (128, 128, 128),  # Gray for metal
    (255, 0, 255),    # Magenta for misc
    (0, 128, 255),    # Orange for net
    (0, 255, 128),    # Light green for plastic bag
    (128, 0, 128),    # Purple for plastic bottle
    (255, 128, 0),    # Light blue for plastic
    (128, 0, 0),      # Dark blue for rod
    (255, 255, 128),  # Light yellow for sunglasses
    (0, 128, 0)       # Dark green for tyre
]

def get_class_name(class_id):
    """Get the class name for a given class ID"""
    if 0 <= class_id < len(EXPECTED_CLASSES):
        return EXPECTED_CLASSES[class_id]
    return f"Unknown_{class_id}"

def get_class_name_short(class_id):
    """Get the short class name for a given class ID"""
    if 0 <= class_id < len(DISPLAY_NAMES):
        return DISPLAY_NAMES[class_id]
    return f"Unknown_{class_id}"

def get_class_color(class_id):
    """Get the color for a given class ID"""
    if 0 <= class_id < len(COLORS):
        return COLORS[class_id]
    return (0, 255, 0)  # Default green

def get_all_classes():
    """Get all class names"""
    return {i: name for i, name in enumerate(EXPECTED_CLASSES)}

def get_all_classes_short():
    """Get all short class names"""
    return {i: name for i, name in enumerate(DISPLAY_NAMES)}

def debug_class_mapping():
    """Debug function to print all class mappings"""
    print("🔍 Debug: Class ID to Name Mapping")
    print("=" * 50)
    for class_id in range(len(EXPECTED_CLASSES)):
        name = get_class_name(class_id)
        short_name = get_class_name_short(class_id)
        color = get_class_color(class_id)
        print(f"Class {class_id}: {name} -> {short_name} (Color: {color})")
    print("=" * 50)

def update_mapping_from_model(model_names):
    """Update mapping based on actual model class names"""
    global EXPECTED_CLASSES, DISPLAY_NAMES
    if model_names:
        print(f"🔄 Updating mapping from model: {model_names}")
        # Extract values from the dictionary and convert to list
        EXPECTED_CLASSES = list(model_names.values())
        # Update display names to match
        DISPLAY_NAMES = [str(name).title() for name in model_names.values()]

# Run debug if this file is executed directly
if __name__ == "__main__":
    debug_class_mapping() 