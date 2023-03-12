#! /bin/bash

wget -P /tmp http://images.cocodataset.org/annotations/annotations_trainval2017.zip && \
unzip /tmp/annotations_trainval2017.zip -d /tmp

python3 <<EOF
import json

with open('/tmp/annotations/instances_val2017.json', 'r') as f:
    data = json.load(f)

with open('./categories_COCO2017.json', 'w') as f:
    json.dump(data['categories'], f, indent=4)
EOF

