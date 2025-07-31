f = open("output.txt", "r")
f_lines = f.readlines()

l_time_sum = 0
l_count = 0
for l in f_lines:
  if l.startswith("Run Time (s): real "):
    l_words = l.split(" ")
    l_time = float(l_words[4])
    l_time_sum += (l_time * 1000)
    l_count += 1

print(l_time_sum/l_count)