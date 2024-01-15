# SAMPLE CODE FROM CHATGPT

import json
import uuid

def create_labelbox_annotation(shape, data_row_id, schema_id_mapping):
    # Assuming the 'points' are top-left and bottom-right for a rectangle
    top_left = shape['points'][0]
    bottom_right = shape['points'][1]
    
    width = bottom_right[0] - top_left[0]
    height = bottom_right[1] - top_left[1]
    
    return {
        'uuid': str(uuid.uuid4()),
        'schemaId': schema_id_mapping.get(shape['label'], 'YOUR_DEFAULT_SCHEMA_ID'),
        'dataRow': {'id': data_row_id},
        'bbox': {
            'top': top_left[1],
            'left': top_left[0],
            'height': height,
            'width': width
        }
    }

def convert_to_labelbox_format(json_data, schema_id_mapping, data_row_id):
    # Convert shapes to Labelbox format
    labelbox_annotations = [create_labelbox_annotation(shape, data_row_id, schema_id_mapping) for shape in json_data['shapes']]

    # The overall structure for Labelbox
    labelbox_data = {
        'annotations': labelbox_annotations
    }

    return labelbox_data

# Schema ID mapping (you need to replace these with actual schema IDs from Labelbox)
schema_id_mapping = {
    'mooring': 'YOUR_SCHEMA_ID_FOR_MOORING'
}

# Data row ID (replace with your actual dataRowId from Labelbox)
data_row_id = 'YOUR_DATAROW_ID'

# Load the original JSON file
file_path = 'path_to_your_json_file.json'
with open(file_path, 'r') as file:
    original_json_data = json.load(file)

# Convert the data to Labelbox format
converted_data = convert_to_labelbox_format(original_json_data, schema_id_mapping, data_row_id)

# Save the converted data to a new JSON file
output_file_path = 'path_to_your_output_json_file.json'
with open(output_file_path, 'w') as file:
    json.dump(converted_data, file, indent=4)

print(f"Converted data saved to {output_file_path}")
