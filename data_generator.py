# -*- coding: utf-8 -*-
"""Data generator

Generate train data

"""

import random
import csv
import math
import json


def main():
    step_id = 10
    roadmap_id_range = range(8, 108)
    library = ""
    roadmap = ""
    step = ""
    dataset = []

    csv_file = open("/workspace/datasets/navict/library.csv", "r")
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)

    library_ids = []
    for row in f:
        library_ids.append(int(row[0]))
        library += "{" + f"id: {row[0]}, title: '{row[1]}', link: '{row[2]}', img: null" + "},"

    # library_ids = [ i for i in range(6, 106)]
    nums = [ i for i in range(0, 10)]
    roadmaps = []
    num_category_unit = 20
    num_pool = math.ceil(len(library_ids) // num_category_unit)

    pools = []
    for i in range(num_pool):
        pool = library_ids[i * num_category_unit: (i+1) * num_category_unit]
        if len(pool) < num_category_unit:
            pool = library_ids[-num_category_unit:]
        pools.append(pool)

    for i in roadmap_id_range:
        pool = pools[i % num_pool]
        l = random.sample( pool, 10)
        l.sort()
        dataset.append(l)

        roadmap += "{" + f"id: {i}, title: 'dummy{i}', firstStepId: {step_id}, userId: 5" + "},"
        for idx, v in enumerate(l):
            step += "{" + f"id: {step_id}, nextStepId: {'null' if idx == 9 else step_id + 1}, roadmapId: {i}, libraryId: {v}" + "},"
            step_id += 1

    print(dataset)

    with open('/workspace/datasets/navict/data.json', mode='wt', encoding='utf-8') as f:
        json.dump(dataset, f)

    with open("/workspace/datasets/navict/library.txt", mode='w') as f:
        f.write(library)

    with open("/workspace/datasets/navict/roadmap.txt", mode='w') as f:
        f.write(roadmap)

    with open("/workspace/datasets/navict/step.txt", mode='w') as f:
        f.write(step)


if __name__ == "__main__":
    main()