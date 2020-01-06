import os

os.chdir("C:\\AutoTracker")
dirlog=open('current_directory1.LOG','r')
path=dirlog.readline().split('"')[1]
basename=dirlog.readline().split('"')[1]
stg_pos=dirlog.readline()

logdir=os.path.join(path,'AutoTrackerLog')
recdir=os.path.join(path,'CARE_Reconstructed')

try:
    os.mkdir(logdir)
except:
    #os.system('del '+os.path.join(logdir,'*.log'))
    pass

try:
    os.mkdir(recdir)
except:
    #os.system('del '+os.path.join(logdir,'*.log'))
    pass

try:
    os.system('del *.ini')
except:
    pass

try:
    os.remove('current_directory.bkp')
except:
    pass

try:
    os.remove('current_directory.LOG')
except:
    pass

# os.chdir("C:\\AutoTracker")
# dirlog=open('current_directory.LOG','r')
# path=dirlog.readline().split('"')[1]
# basename=dirlog.readline().split('"')[1]
# stg_pos=dirlog.readline()
#
# INI_out= 'AutoTracker_Stg'+stg_pos+'.INI'
# log_out = 'AutoTracker_Stg'+stg_pos+'.log'
#
# try:
#     os.remove(INI_out)
# except:
#     pass
#
# try:
#     os.remove(log_out)
# except:
#     pass
