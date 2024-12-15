import collections

cf_file = 'datasets/ml-1m/train.txt'
kg_file = 'datasets/ml-1m/kg_final.txt'
save_file = 'datasets/ml-1m/cooccurrence.txt'

nodes = collections.defaultdict(list)
items = set()
# load cf_data
lines = open(cf_file, 'r').readlines()
for line in lines:
    temp = [int(x) for x in line.split()]
    items.add(temp[1])
lines = open(cf_file, 'r').readlines()
for line in lines:
    temp = [int(x) for x in line.split()]
    if temp[1] not in nodes.keys():
        nodes[temp[1]] = [temp[0]]
    else:
        nodes[temp[1]].append(temp[0])
# load kg_data
h = []
r = []
t = []
lines = open(kg_file, 'r').readlines()
for line in lines:
    temp = [int(x) for x in line.split()]
    assert len(temp) == 3
    h.append(temp[0])
    r.append(temp[1])
    t.append(temp[2])
    if temp[0] in nodes.keys():
        nodes[temp[0]].append(temp[2])
    if temp[2] in nodes.keys():
        nodes[temp[2]].append(temp[0])

assert len(h) == len(r)
assert len(h) == len(t)
n_relations = len(set(r))
n_h_origin = len(h)
print(n_h_origin)
print(n_relations)
coorrence_h = collections.defaultdict(list)
item_list = list(nodes.keys())
intersection_num = 0
for i in range(len(item_list)):
    target_set = set(nodes[item_list[i]])
    for j in range(i + 1, len(item_list)):
        object_set = set(nodes[item_list[j]])
        if len(target_set.intersection(object_set)) > 3:
            coorrence_h[item_list[i]].append(item_list[j])
            coorrence_h[item_list[j]].append(item_list[i])
            intersection_num += 1
print(intersection_num)  # 2299 15,518
writer = open(save_file, 'w')
# assert len(h) == len(r)
# assert len(h) == len(t)
# assert len(h) == intersection_num + n_h_origin
for k in coorrence_h:
    line = "" + str(k) + " "
    for v in coorrence_h[k]:
        line += str(v)
        line += " "
    line += "\n"
    writer.write(line)

writer.close()
