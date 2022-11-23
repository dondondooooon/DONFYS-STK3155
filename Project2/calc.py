import  numpy as np

n = 569
batch_size = 30
batch_total = []
for i in range(n):
    batch_total.append(i+1)

total_b_30 = np.array_split(batch_total, batch_size)

b_1 = total_b_30[0]
b_25 = total_b_30[24]
b_30 = total_b_30[29]
print("569 /30 = ", (569/30))

print("\nfirst batch first element", b_1[0])
# print("first batch last element", b_1[568])

print("\nlast batch first element", b_30[0])
# print("last batch last element", b_30[568])

print("\nbatch 25 first element", b_25[0])
# print("batch 25 last element", b_25[568])

print("\n")