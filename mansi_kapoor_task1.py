import sys
import time
import math
from itertools import combinations
from pyspark import SparkContext

sc = SparkContext()

startTime = time.time()
#reading command line arguments
case = int(sys.argv[1])
supportValue = int(sys.argv[2])
input_path = sys.argv[3]
output_path = sys.argv[4]
#reading the input file
raw_data = sc.textFile(input_path)
#making baskets according to the case
if case == 1:
    all_baskets = raw_data.map(lambda line: line.split(',')).filter(lambda line: line[0] != 'user_id').map(lambda line: (line[0],[line[1]])).reduceByKey(lambda x,y: x+y).map(lambda line: list(set(line[1])))

elif case == 2:
    all_baskets = raw_data.map(lambda line: line.split(',')).filter(lambda line: line[0] != 'user_id').map(lambda line: (line[1],[line[0]])).reduceByKey(lambda x,y: x+y).map(lambda line: list(set(line[1])))
#number of partitions of data
partitions = all_baskets.getNumPartitions()


def caseone(frequent_items,k):
    lst = []
    for x in range(len(frequent_items)):
            for y in range(x+1, len(frequent_items)):
                pair = frequent_items[x] | frequent_items[y]
                if len(pair) == k:
                    if pair in lst:
                        continue
                    lst.append(pair)
    return lst



def case2(frequent_items,k):
    lst = []

    for x in range(len(frequent_items)):
            for y in range(x+1, len(frequent_items)):
                common_el = frequent_items[x].intersection(frequent_items[y])
                if len(common_el) == k-2:
                    pair = frequent_items[x] | frequent_items[y]
                    if pair in lst:
                        continue
                    pairs = list(combinations(pair, k-1))
                    c = 0
                    for i in pairs:
                        if c == k-2:
                            break
                        if not common_el.issubset(i):
                            if set(i) in frequent_items:
                                c += 1
                    if c == k-2:
                        lst.append(pair)
    return lst

#function to compute candidate itemsets in each partition
def compute_candidates(frequent_items,k):
    lst = []
    if k == 2:
       
        lst=caseone(frequent_items,k)

    else:
        lst = case2(frequent_items,k)

    return lst

#phase 1


def partitionFunction(d,supportForOnePartition):
    frequent_items = []

    for item in d.keys():
        if d[item] >= supportForOnePartition:
            frequent_items.append(item)
    return frequent_items

#Apriori algorithm


def apriori(i):
    baskets = list(i)
    supportForOnePartition = math.ceil(float(supportValue)/partitions) #support value for each partition
    k = 2
    d = {}
    frequent_items = []
    frequent_itemset = []

    b = 0
    while b < len(baskets):
        for i in baskets[b]:
            d[i] = d.get(i,0) + 1
        b += 1

    frequent_items=partitionFunction(d,supportForOnePartition)

    frequent_itemset = [(i,1) for i in frequent_items]
    frequent_items = [{i} for i in frequent_items]

    number_of_FI = len(frequent_items)
    
    while (number_of_FI > 0):
        candidate_set = compute_candidates(frequent_items,k)
        frequent_items = []
        a = 0
        while a < len(candidate_set):
            c = 0
            for basket in baskets:
                if candidate_set[a].issubset(basket):
                    c += 1
            if c >= supportForOnePartition:
                frequent_items.append(candidate_set[a])
            a += 1

        number_of_FI = len(frequent_items)
        if number_of_FI > 0:
            frequent_itemset += [(tuple(i),1) for i in frequent_items]

        k = k + 1

    return(frequent_itemset)
#getting candidate itemsets
candidate_items = all_baskets.mapPartitions(apriori).groupByKey().map(lambda x:x[0]).collect()

for x in range(len(candidate_items)):
    if not isinstance(candidate_items[x], tuple):
            candidate_items[x] = (candidate_items[x],)


l_max = 0
for x in candidate_items:
    if len(x) > l_max:
        l_max = len(x)
#writing into dictionary for output file
candidates = {x:[] for x in range(1, l_max+1)}
for x in candidate_items:
    candidates[len(x)].append(tuple(sorted(x)))

#phase 2
#function to calculate actual counts of candidate itemsets
def FilePointer(output_path,candidates,final_frequent):
    with open(output_path, 'w') as fp:
        fp.write('Candidates:\n')
        for k in candidates:
            candidates[k] = list(set(candidates[k]))
            candidates[k].sort()
            s = str(candidates[k]).replace(',)', ')')
            fp.write(s[1:-1])
            fp.write('\n\n')

        fp.write('Frequent Itemsets:\n')
        for k in final_frequent:
            final_frequent[k] = list(set(final_frequent[k]))
            final_frequent[k].sort()
            s = str(final_frequent[k]).replace(',)', ')')
            fp.write(s[1:-1])
            fp.write('\n\n')

    fp.close()

def count_all(i,candidate_items):
    basket = list(i)
    loc_candidate = []
    a = 0
    while a < len(candidate_items):
        c = 0
        if not isinstance(candidate_items[a], tuple):
            candidate_items[a] = (candidate_items[a],)
        b = 0
        while b < len(basket):
            check = all(item in basket[b] for item in candidate_items[a])
            if check == True:
                c = c + 1
            b += 1

        loc_candidate.append((candidate_items[a],c))
        a += 1

    return(loc_candidate)

candidate_counts = all_baskets.mapPartitions(lambda x: count_all(x,candidate_items)).reduceByKey(lambda x,y: x+y).collect()

frequent_items = []
l_max = 0
#filtering out actual frequent items
for x in candidate_counts:
    if x[1] >= supportValue:
        frequent_items.append(x[0])
        if len(x[0]) > l_max:
            l_max = len(x[0])

#writing into dictionary for output file
final_frequent = {x:[] for x in range(1, l_max+1)}
for x in frequent_items:
    final_frequent[len(x)].append(tuple(sorted(x)))

#writing into output file
FilePointer(output_path,candidates,final_frequent)

endTime = time.time()
print("\nDuration: {}".format(endTime - startTime))
