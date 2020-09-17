import os
import re

def sort_key(s):
    # 排序关键字匹配
    # 匹配开头数字序号
    if s:
        try:
            c = re.findall('^\d+', s)[0]
        except:
            c = -1
        return int(c)

def strsort(alist):
    alist.sort(key=sort_key)
    return alist
path=r'H:\lidongyv.github.io\assets\img\project7'
files=os.listdir(path)
files=strsort(files)
for i in range(len(files)):
    files[i].split('幻灯片')
    print(i,files[i],files[i].split('幻灯片'))
    os.rename(os.path.join(path,files[i]),os.path.join(path,str(i)+'.png'))
    # os.rename(os.path.join(path,files[i]),os.path.join(path,files[i].split('幻灯片')[1]))