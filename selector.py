import numpy as np
import collections
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm

file_name = 'datasets/ml-1m/'
save_file = file_name + 'longtail2.txt'


def load_cf(filename):
    users = list()
    items = list()
    user_dict = dict()
    lines = open(filename, "r").readlines()

    for l in lines:
        temp = l.strip()
        interaction = [int(i) for i in temp.split()]
        if len(interaction) > 1:
            user_id, item_ids = interaction[0], interaction[1:]
            # deduplicate
            item_ids = list(set(item_ids))
            for item_id in item_ids:
                users.append(user_id)
                items.append(item_id)
            user_dict[user_id] = item_ids

    return users, items, user_dict


def load_kg(filename, n_items):
    h = set()
    r = set()
    t = set()
    triples = set()
    # No relation
    kg_dict = collections.defaultdict(list)
    entitytoitem_dict = collections.defaultdict(list)
    itemtoentity_dict = collections.defaultdict(list)

    for l in open(filename, "r").readlines():
        temp = l.strip()
        triple = [int(i) for i in temp.split()]
        assert len(triple) == 3
        h.add(triple[0])
        r.add(triple[1])
        t.add(triple[2])
        triples.add((triple[0], triple[1], triple[2]))
        kg_dict[triple[0]].append(triple[2])

        if triple[0] >= n_items and triple[2] < n_items:
            entitytoitem_dict[triple[0]].append(triple[2])
            itemtoentity_dict[triple[2]].append(triple[0])
        if triple[0] < n_items and triple[2] >= n_items:
            itemtoentity_dict[triple[0]].append(triple[2])
            entitytoitem_dict[triple[2]].append(triple[0])
    return h, r, t, triples, kg_dict, entitytoitem_dict, itemtoentity_dict


def get_item_dict(user_dict):
    item_dict = collections.defaultdict(list)
    for user in user_dict:
        for item in user_dict[user]:
            item_dict[item].append(user)
    return item_dict


cf_file = file_name + 'train.txt'
test_file = file_name + 'test.txt'
kg_file = file_name + 'kg_final.txt'
num = 2

users, items, user_dict = load_cf(cf_file)
item_dict = get_item_dict(user_dict)
n_users = max(users) + 1
n_items = max(items) + 1
# test_users, test_items, test_user_dict = load_cf(test_file)
h, r, t, triples, kg_dict, entitytoitem_dict, itemtoentity_dict = load_kg(kg_file, n_items)

n_entities = max(max(h), max(t)) + 1

pretrain_embed_dir = 'datasets/pretrain/ml-1m/selector.npz'
pretrain_embed = np.load(pretrain_embed_dir)
entity_user_embed = pretrain_embed['entity_user_embed']
print(entity_user_embed.shape[0], n_users, n_items, n_users + n_items)
assert entity_user_embed.shape[0] == (n_users + n_entities)

entity_user_embed = torch.tensor(entity_user_embed)
writer = open(save_file, 'w')
for user_id in tqdm(user_dict):
    interacted_items = user_dict[user_id]
    userid_shift = user_id + n_entities
    user_embed = entity_user_embed[[userid_shift]]
    user_embed = user_embed.unsqueeze(2)
    favorite_entities = set()
    for interacted_item in interacted_items:
        interacted_item_embed = entity_user_embed[[interacted_item]]
        candidate_entities = itemtoentity_dict[interacted_item]
        candidate_entities_embed = entity_user_embed[candidate_entities]
        candidate_entities_embed = candidate_entities_embed.unsqueeze(1)
        p_entity = interacted_item_embed * candidate_entities_embed
        p_entity = F.softmax(p_entity, dim=2)
        p = torch.matmul(p_entity, user_embed)
        p = p.squeeze()
        # 逆序(从大到小)
        indices = np.argsort(-p)
        assert len(indices) == len(candidate_entities)
        favorite_entities.add(candidate_entities[indices[0]])
    long_tail_items = set()
    for favorite_entity in favorite_entities:
        favorite_entity_embed = entity_user_embed[[favorite_entity]]
        candidate_items = entitytoitem_dict[favorite_entity]
        candidate_items_embed = entity_user_embed[candidate_items]
        candidate_items_embed = candidate_items_embed.unsqueeze(1)
        popularity_attention = list()
        for candidate_item in candidate_items:##  (math.log(len(item_dict[candidate_item])) + len(item_dict[candidate_item]))
            ## (math.log(len(item_dict[candidate_item]), 350) + 0.001)
            pop = len(item_dict[candidate_item])
            if pop == 0:
                pop = 350
            popularity_attention.append(1 / (math.log(pop,350) + 0.001))

        popularity_attention = torch.tensor(popularity_attention)
        p_item = favorite_entity_embed * candidate_items_embed
        p_item = F.softmax(p_item, dim=2)
        p = torch.matmul(p_item, user_embed)
        p = p.squeeze()
        p = popularity_attention * p

        indices = np.argsort(-p)
        assert len(indices) == len(candidate_items)
        if len(candidate_items)>num:
            for i in range(num):
                long_tail_items.add(candidate_items[indices[i]])
        else:
            long_tail_items.add(candidate_items[indices[0]])
    long_tail_items = sorted(long_tail_items)
    strline = '' + str(user_id) + ' '
    for item in long_tail_items:
        strline = strline + str(item) + ' '
    strline += '\n'
    writer.write(strline)
