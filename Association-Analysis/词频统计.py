from collections import defaultdict

data = [
    ['r', 'z', 'h', 'j', 'p'],
    ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
    ['z'],
    ['r', 'x', 'n', 'o', 's'],
    ['y', 'r', 'x', 'z', 'q', 't', 'p'],
    ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']
]

d = defaultdict(int)

for item in data:
    for one in item:
        d[one] += 1

filter_d = {k: v for k, v in d.items() if v >= 3}
print(filter_d)

result = sorted(filter_d.items(), key=lambda p: p[1], reverse=True)
print(result)
