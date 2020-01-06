import os
import sys

os.chdir("C:\\AutoTracker")
try:
    os.system('del '+sys.argv[1])
except:
    pass
