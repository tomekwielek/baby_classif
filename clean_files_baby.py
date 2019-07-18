#NOTICE script deletes al filles older than delta (days) in folders \
# NOT being in exclude and ending woth 'endswith'

#OR all emtpy directories
import os, time, datetime
import os.path as osp
from IPython.core.debugger import set_trace
path = 'C:\\Users\\b1016533\\Desktop\\data_plos\\subjects\\'
#path = osp.join(base_path, 'wake') #arg

listdir = os.listdir(path)

exclude = ['psd', 'mspet1m3']

delta = 1 #days
endswith = '.txt'
ok_time = datetime.datetime.now() - datetime.timedelta(days=delta)
ok_time = (ok_time - datetime.datetime(1970, 1, 1)).total_seconds()

for (root, dirs, files) in os.walk(path, topdown=True):
    #modify dirs inplace, include all 'VP' folders
    dirs[:] = [d for d in dirs if d not in exclude]
    '''
    for filename in files:
       if filename.endswith(endswith) :
           thefile = osp.join(root,filename)
           #modification time
           file_time = osp.getmtime(thefile)
           if file_time < ok_time:
              print(time.ctime(file_time))
              print(osp.join(root,filename))
              #set_trace()
              # uncomm. if sure to use it!!!!
              os.remove(osp.join(root,filename))
              #set_trace()
    '''
    #delete empty dirs
    if len(os.listdir(root) ) == 0:
        print('Directory is empty')
        os.rmdir(root)
