def apriori_gen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):
            L1 = list(Lk[i])[:k - 2]
            L2 = list(Lk[j])[:k - 2]
            print(L1)
            print(L2)
            L1.sort()
            L2.sort()
            if L1 == L2:
                print(Lk[i])
                print(Lk[j])
                retList.append(Lk[i] | Lk[j])

                print(retList)
    return retList


L0 = [{0}, {1}, {2}, {3}]
print(apriori_gen(L0, 2))
print('-'*20)

L1 = [{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}]
print(apriori_gen(L1, 3))

L2 = [{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}]
print(apriori_gen(L2, 4))
