def apriori_gen(frequent_set_k, k):
    candidate_list = []
    len_frequent_set_k = len(frequent_set_k)
    for i in range(len_frequent_set_k):
        for j in range(i + 1, len_frequent_set_k):

            # print(frequent_set_k[i], frequent_set_k[j])
            L1 = list(frequent_set_k[i])[:k - 2]
            L2 = list(frequent_set_k[j])[:k - 2]
            # print(L1)
            # print(L2)
            L1.sort()
            L2.sort()
            if L1 == L2:
                # print(frequent_set_k[i])
                # print(frequent_set_k[j])
                candidate_list.append(frequent_set_k[i] | frequent_set_k[j])

                # print(candidate_list)
    return candidate_list


#
L0 = [{0}, {1}, {2}, {3}]
print(apriori_gen(L0, 2))
print('-' * 20)

L1 = [{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}]
print(apriori_gen(L1, 3))

L2 = [{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}]
print(apriori_gen(L2, 4))

L2 = [{5, 3}, {5, 10}]
print(apriori_gen(L2, 3))
