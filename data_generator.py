# -*- coding: utf-8 -*-
"""Data generator

Generate train data

"""

import random


def main():
    library_ids = [ i for i in range(6, 106)]
    nums = [ i for i in range(0, 10)]
    roadmap_id_range = range(8, 108)
    step_id = 10

    roadmap = ""
    step = ""

    for i in roadmap_id_range:
        l = random.sample(library_ids, 10)
        l.sort()
        shuffle_ids = random.sample(nums, 2)
        tmp = l[shuffle_ids[0]]
        l[shuffle_ids[0]] = l[shuffle_ids[1]]
        l[shuffle_ids[1]] = tmp

        roadmap += "{" + f"id: {i}, title: 'dummy{i}', firstStepId: {step_id}, userId: 5" + "},"
        for idx, v in enumerate(l):
            step += "{" + f"id: {step_id}, nextStepId: {'null' if idx == 9 else step_id + 1}, roadmapId: {i}, libraryId: {v}" + "},"
            step_id += 1

    with open("roadmap.txt", mode='w') as f:
        f.write(roadmap)

    with open("steptxt", mode='w') as f:
        f.write(step)


if __name__ == "__main__":
    main()