import os
import json

with open("benchmarking.json", "r") as f:
    benchmark = json.load(f)
    
with open("S1.json", "r") as f:
    map = json.load(f)

# go through each key in the benchmark and find the correct ID in the map and add the value of the map to the benchmark
keys = list(benchmark.keys())
keys_map = list(map.keys())
for key in keys:
    id = benchmark[key][0]
    if id in keys_map:
        benchmark[key].append(map[id])
    else:
        print(f"ID {id} not found in map")

# write the benchmark to a new file
with open("benchmarking_mapped.json", "w") as f:
    json.dump(benchmark, f, indent=4)