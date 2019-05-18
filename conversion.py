
"""Converts all .au files into .wav in provided directory
using SOX converter
Example of running this code:
python conversion.py path_1, path_2, ..., path_n
path_i is FULL path to directory
NOTE: all files will be replaced with theirs .wav copies
If you want to keep .au files copy and save them before using this script
"""

import sys
import os

genre_dirs = sys.argv[1:]

for genre_dir in genre_dirs:
    
    os.chdir(genre_dir)

    print('Contents of ' + genre_dir + ' before conversion: ')
    os.system("ls")

    for file in os.listdir(genre_dir):
        os.system("sox " + str(file) + " " + str(file[:-3]) + ".wav")
    
    os.system("rm *.au")
    print('After conversion:')
    os.system("ls")
    print('\n')

print("Conversion complete. Check respective directories.")
