dict1 = {'Name': 'Zara', 'Age': 7}
dict2 = {'Sex': 'female', 'Name': 'liwei', 'Age': 24}

# 如果 dict2 中有和 dict1 的 key 相同的 ，用 dict2 的 value 去覆盖它。
dict1.update(dict2)

print(dict1)
