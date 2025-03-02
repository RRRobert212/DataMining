import math

sum = 0
for i in range(13,26):
    sum += math.comb(25, i) * 0.45**i *0.55**(25-i)
    print(sum)

print("Total: " + str(sum))