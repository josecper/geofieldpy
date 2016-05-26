import random
import argparse

parser = argparse.ArgumentParser(description="turn part of a dif file into nans")
parser.add_argument("--density", type=float, default=0.1)
parser.add_argument("--filename")
parser.add_argument("--out")
parser.add_argument("--columns", type=int, nargs="+", default=[2, 3, 4])
parser.add_argument("--nan", default="NaN")

namespace = parser.parse_args()

density = namespace.density
filename = namespace.filename
out = namespace.out
columns = namespace.columns
nan = namespace.nan

f_in=open(filename,"r")
f_out=open(out,"w")

for line in f_in:
    if line.startswith("#"):
        f_out.write(line)
    else:
        values = line.split("\t")
        new_values = [nan if (random.random() < density) and (i in columns) else v for i,v in enumerate(values)]
        f_out.write(" ".join(new_values)+"\n")

f_in.close()
f_out.close()
