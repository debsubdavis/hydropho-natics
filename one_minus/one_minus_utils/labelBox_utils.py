# SAMPLE CODE FROM CHATGPT

import json
import uuid

def create_labelbox_annotation(shape, data_row_id, schema_id_mapping):
    # Assuming the 'points' are top-left and bottom-right for a rectangle
    start = shape['points'][0]
    stop = shape['points'][1]
    
    #   "shapes": [
    # {
    #   "label": "mooring",
    #   "points": [
    #     [
    #       322.18518518518516, # top left
    #       592.1111111111111
    #     ],
    #     [
    #       388.85185185185185, # bottom right
    #       737.2962962962962
    #     ]
    #   ],

    1
# [[536.7831325301204, 1030.1445783132529], [965.6987951807228, 562.6746987951807]]
# [[1007.867469879518, 526.5301204819276], [1072.9277108433735, 643.3975903614457]]
# [[1082.5662650602408, 527.7349397590361], [1157.2650602409637, 648.2168674698794]]
# [[1369.313253012048, 948.2168674698794], [1431.9638554216867, 624.1204819277108]]
# [[1497.024096385542, 968.6987951807228], [1553.6506024096384, 765.0843373493975]]
# [[1606.6626506024095, 978.3373493975903], [1569.313253012048, 606.0481927710842]]
# [[1636.7831325301204, 975.9277108433735], [1690.9999999999998, 860.2650602409637]]
# [[1711.4819277108431, 875.9277108433735], [1756.0602409638552, 1000.024096385542]]
# [[1776.5421686746986, 951.8313253012047], [1818.7108433734938, 868.6987951807228]]
# [[1856.0602409638552, 855.4457831325301], [1886.1807228915661, 959.0602409638552]]
# [[1912.6867469879517, 951.8313253012047], [1959.6746987951806, 839.7831325301204]]
# [[2086.180722891566, 542.1927710843373], [2139.192771084337, 602.433734939759]]

    # bbox_with_radio_subclass_annotation = lb_types.ObjectAnnotation(
    # name="bbox_with_radio_subclass",
    # value=lb_types.Rectangle(
    #     start=lb_types.Point(x=541, y=933),  # x = left, y = top 
    #     end=lb_types.Point(x=871, y=1124),  # x= left + width , y = top + height
    # ),
    # classifications=[
    #     lb_types.ClassificationAnnotation(
    #         name="sub_radio_question",
    #         value=lb_types.Radio(answer=lb_types.ClassificationAnswer(
    #             name="first_sub_radio_answer")))
    # ])
    
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

def need_to_fix():
    # {'mooring', 'boat', 'airplane', 'helicopter', 'fish', 'humpback', 'flow noise', 'mooring noise'}
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
