#FOLDER NAMED DATASET MUST CONTAIN THE TEST.JSON
#FOLDER NAMED OUTPUT MUST CONTAIN ALL THE TXT FILES FROM YOLO

import glob
import json

result = []

with open("dataset/test.json") as f:
  data = json.load(f)
image_id = dict()
for i in data['images']:
    image_id[i['file_name'].split('.')[0]] = i['id']  #[:-4] to remove extensions
# print(image_id)
for i in (glob.glob("output/*.txt")):
#    print(i[7:])
    open_file = open(i,'r')
    for j in open_file.readlines():
        arr=j.split(" ")
        print(arr[-1].split('\n'))
        arr[-1] = arr[-1].split('\n')[0]
        new_result = dict()
        new_result['image_id'] = int(image_id[i[7:].split('.')[0]]) #to read outputfile and remove extensions
        new_result['category_id'] = int(arr[0])
        new_result['score'] = float(arr[1])
        new_result['bbox'] = [float(arr[i]) for i in range(2,6)]
        result.append(new_result)
# print(result)
out_file = open("result.json", "w")  
    
json.dump(result, out_file, indent = 6)  
    