import json

def get_dict_schema(data, parent_key=''):
    schema = {}
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            sub_schema = get_dict_schema(value, new_key)
            schema.update(sub_schema)
    elif isinstance(data, list):
        if data:
            list_item_schema = get_dict_schema(data[0], parent_key)
            schema[parent_key] = [list_item_schema]
    else:
        schema[parent_key] = type(data).__name__
    return schema


# Example dictionary for testing
with open("datasets/coco_dataset/instances_val2017.json", "r") as f:
    sample_dict = json.load(f)

# schema = get_dict_schema(sample_dict)
#
# # Print the schema
# for key, value in schema.items():
#     print(f"{key}: {value}")


def get_dict_schema_2(data):
    stack = [(data, '')]
    schema = {}
    while stack:
        current_dict, parent_key = stack.pop()
        if isinstance(current_dict, dict):
            for key, value in current_dict.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                stack.append((value, new_key))
        elif isinstance(current_dict, list):
            if current_dict:
                list_item = current_dict[0]
                if not isinstance(list_item, list) and not isinstance(list_item, dict) and len(current_dict) <= 5:
                    for index, item in enumerate(current_dict):
                        item_key = f"{parent_key}.{index}"
                        stack.append((item, item_key))
                else:
                    list_item_key = f"{parent_key}.index"
                    stack.append((list_item, list_item_key))
        else:
            schema[parent_key] = type(current_dict).__name__
    return schema


# schema = get_dict_schema_2([sample_dict])
#
# # Print the schema
# for key, value in schema.items():
#     print(f"{key}: {value}")
#
# print(sample_dict.keys())
# print(schema)

i = ".asdasdsas."
b = i.removeprefix(".asdasdsas.")
print(b)
print(bool(b))
