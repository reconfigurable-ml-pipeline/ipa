import json
import numpy as np

# def convert_values_to_strings(dictionary):
#     new_dict = {}
#     for key, value in dictionary.items():
#         if isinstance(value, dict):
#             new_value = convert_values_to_strings(value)
#         else:
#             new_value = str(value)
#         new_dict[key] = new_value
#     return new_dict


class Int64Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)
