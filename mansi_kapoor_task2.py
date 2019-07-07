import sys
from pyspark import SparkContext
import time
import itertools

start = time.time()
sc = SparkContext()

threshold = int(sys.argv[1])
support = int(sys.argv[2])
input_data_file = sys.argv[3]
out_file = sys.argv[4]


# counting frequency of itemset
def count_frequency(itemset, superset):
    count = 0
    for l in superset:
        if (set(itemset).issubset(l)):
            count = count + 1
    return count


def count_frequency_SONPhase2(baskets_list, itemsets):
    baskets = list(baskets_list)

    frequency_value = list()

    for i in itemsets:
        count = 0
        for b in baskets:
            if (set(i).issubset(b)):
                count = count + 1
        frequency_value.append([i, count])

    return frequency_value

# apriori alogorithm

def apriori(baskets_list, itemset, support_threshold):
    baskets = list(baskets_list)
    size = 1;

    frequent_itemsets = list()
    C = []
    L = []

    k = 1

    for i in itemset:
        count = 0
        for b in baskets:
            if (i in b):
                count = count + 1

        C.append(i)
        if (count >= support_threshold):
            L.append(i)

    C.sort()
    L.sort()
    length_l1 = len(L)

    L1 = [(x,) for x in L]
    frequent_itemsets.extend(L1)

    k = k + 1

    # genearting pairs
    C = list()
    for x in itertools.combinations(L, 2):
        pair = list(x)
        pair.sort()
        C.append(pair)
    C.sort()

    L.clear()
    for c in C:
        count = count_frequency(c, baskets)
        if (count >= support_threshold):
            L.append(c)

    L.sort()

    frequent_itemsets.extend(L)
    k = k + 1

    while k != length_l1:

        C.clear()
        C = list()

        for i in range(len(L) - 1):
            for j in range(i + 1, len(L)):
                if (L[i][0:k - 2] == L[j][0:k - 2]):
                    candidate = list(set(L[i]) | set(L[j]))
                    candidate.sort()
                    if (candidate not in C):
                        C.append(candidate)
                else:
                    break

        if (len(C) == 0):
            break
        C.sort()

        L.clear()
        for c in C:
            count = count_frequency(c, baskets)
            if (count >= support_threshold):
                L.append(c)
        L.sort()

        frequent_itemsets.extend(L)
        k = k + 1

    return frequent_itemsets

def writeFile():
    # write to output file
    f = open(out_file, 'w')

    # writing the candidates
    f.write("Candidates:")
    f.write("\n")
    l = 1
    while l != len(list_of_items):
        s = ""
        for r in reducePhaseOutput:
            if (len(r) == l):
                s = s + str(r)

        s = s.replace(",)", ")").replace(")(", "),(")
        if (s == ""):
            break
        else:
            if (l != 1):
                f.write("\n\n")
            f.write(s)
        l = l + 1
    f.write("\n\n")

    # writing the frequent itemsets
    f.write("Frequent Itemsets:")
    f.write("\n")
    l = 1
    while l != len(list_of_items):
        s = ""
        for b in SONPhase2Reduce:
            if (len(b[0]) == l):
                s = s + str(b[0])
        s = s.replace(",)", ")").replace(")(", "),(")
        if (s == ""):
            break
        else:
            if (l != 1):
                f.write("\n\n")
            f.write(s)
        l = l + 1
    f.close()

if __name__ == '__main__':

    # reading data
    data = sc.textFile(input_data_file)
    rdd = data.map(lambda x: x.split(',')).map(lambda a: (a[0], a[1])).distinct()
    baskets = rdd.map(lambda a: (a[0], [a[1]])).reduceByKey(lambda a, b: a + b).persist().filter(
        lambda x: x[0] != "user_id").map(lambda x: set(x[1])).filter(lambda x: len(x) > threshold)

    baskets_list = baskets.collect()

    list_of_items_in_baskets = baskets.collect()
    list_of_items = list(set().union(*list_of_items_in_baskets))

    # SON algorithm
    num_partitions = data.getNumPartitions()
    support_threshold = support / num_partitions

    # first phase
    SONPhase1Map = baskets.mapPartitions(lambda b: apriori(b, list_of_items, support_threshold)).map(
        lambda x: (tuple(x), 1))
    reducePhaseOutput = SONPhase1Map.distinct().sortByKey().map(lambda a: a[0]).collect()

    # second phase
    SONPhase2Map = baskets.mapPartitions(lambda b: count_frequency_SONPhase2(b, reducePhaseOutput))
    SONPhase2Reduce = SONPhase2Map.reduceByKey(lambda a, b: (a + b)).filter(
        lambda a: a[1] >= support).sortByKey().collect()

    writeFile()
    end = time.time()
    print("Duration: " + str(end - start))