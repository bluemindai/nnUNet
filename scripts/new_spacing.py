import json
import sys

plans_filepath = sys.argv[1]
batch_size = sys.argv[2]

with open(plans_filepath, 'r') as f:
  j = json.load(f)

j['configurations'].update(
{
  "3d_fullres_custom": {
    "inherits_from": "3d_fullres",
    "batch_size": int(batch_size),
    "spacing": [1.5, 1.5, 1.5],
    "data_identifier": "nnUNetPlans_3d_fullres_custom"
  }
}
)

with open(plans_filepath, 'w') as f:
  json.dump(j, f)