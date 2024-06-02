import json
import random


def get_rand_list(file_path, edit_num, seed):
    # Open and load the JSON file
    with open(file_path, 'r') as f:
        dataset = json.load(f)
    print(len(dataset))
    # you can use seed = 100
    random.seed(seed)
    caseids = [d['case_id'] for d in dataset]
    return random.sample(caseids, edit_num)


file_path_T = 'datasets/MQuAKE-Remastered-T.json'
file_path_3k = 'datasets/MQuAKE-Remastered-CF-3k.json'
file_path_9k = 'datasets/MQuAKE-Remastered-CF-9k.json'
file_path_3151 = 'datasets/MQuAKE-Remastered-CF-3151.json'
file_path_6334 = 'datasets/MQuAKE-Remastered-CF-6334.json'


rand_lists = {
    "rand_list_T_1": get_rand_list(file_path_T, 1, 100),
    "rand_list_T_100": get_rand_list(file_path_T, 100, 100),
    "rand_list_T_500": get_rand_list(file_path_T, 500, 100),
    "rand_list_T_all": get_rand_list(file_path_T, 1864, 100),

    "rand_list_3k_1": get_rand_list(file_path_3k, 1, 100),
    "rand_list_3k_100": get_rand_list(file_path_3k, 100, 100),
    "rand_list_3k_1000": get_rand_list(file_path_3k, 1000, 100),
    "rand_list_3k_all": get_rand_list(file_path_3k, 3000, 100),

    "rand_list_9k_1": get_rand_list(file_path_9k, 1, 100),
    "rand_list_9k_1000": get_rand_list(file_path_9k, 1000, 100),
    "rand_list_9k_3000": get_rand_list(file_path_9k, 3000, 100),
    "rand_list_9k_6000": get_rand_list(file_path_9k, 6000, 100),
    "rand_list_9k_all": get_rand_list(file_path_9k, 9171, 100),

    "rand_list_3151_1": get_rand_list(file_path_3151, 1, 100),
    "rand_list_3151_100": get_rand_list(file_path_3151, 100, 100),
    "rand_list_3151_1000": get_rand_list(file_path_3151, 1000, 100),
    "rand_list_3151_all": get_rand_list(file_path_3151, 3151, 100),
}

