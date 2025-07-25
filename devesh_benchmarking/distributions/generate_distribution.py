import matplotlib.pyplot as plt
import csv

def process_file(filename):
  f = open(filename, "r")
  f_lines = f.readlines()
  lengths = []
  for i in range(1,len(f_lines)):
    lengths.append(len(f_lines[i]))
  
  plt.hist(lengths, bins=15, edgecolor='black')
  plt.savefig("help.pdf")
