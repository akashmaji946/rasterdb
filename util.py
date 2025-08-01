f = open("output.txt", "r")
f_lines = f.readlines()

l_times = []
l_count = 0
for l in f_lines:
  if l.startswith("Run Time (s): real "):
    l_words = l.split(" ")
    l_time = float(l_words[4])
    l_times.append(l_time * 1000)
    l_count += 1

    if l_count == 21:
      l_count = 0
      print("cold time: " + str(l_times[0]))
      right_list = l_times[1:]
      print(sum(right_list)/len(right_list))
      l_times = []