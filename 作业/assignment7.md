# Assignment #7: April 月考

Updated 1557 GMT+8 Apr 3, 2024

2024 spring, Complied by ==狄晨阳 生命科学学院==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：Windows11

Python编程环境：Spyder IDE 5.4.3

C/C++编程环境：无



## 1. 题目

### 27706: 逐词倒放

http://cs101.openjudge.cn/practice/27706/



思路：按空格分割，倒序列表后再输出



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:17:30 2024

@author: 20311
"""

string=input().strip().split()
string=reversed(string)
print(' '.join(string))

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240406102000138](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240406102000138.png)



### 27951: 机器翻译

http://cs101.openjudge.cn/practice/27951/



思路：用一个列表来储存内存中的数据，若达到上限则更新一次，另用一个变量累计查询次数



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:23:19 2024

@author: 20311
"""

m,n=map(int,input().split())
article=list(map(int,input().strip().split()))
t=0
memory=[]
l=0
for word in article:
    if word in memory:
        continue
    else:
        if l==m:
            memory.pop(0)
            t+=1
            memory.append(word)
        else:
            memory.append(word)
            t+=1
            l+=1
print(t)

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240406103040320](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240406103040320.png)



### 27932: Less or Equal

http://cs101.openjudge.cn/practice/27932/



思路：一直WA，没考虑k=0时的特殊情况，后来考虑了之后就好了



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 10:40:44 2024

@author: 20311
"""

n,k=map(int,input().strip().split())
nums=list(map(int,input().strip().split()))
nums.sort()
if k>n:
    print(-1)
elif k==n:
    print(nums[-1])
elif k==0:
    if nums[0]!=1:
        print(1)
    else:
        print(-1)
else:
    if nums[k-1]==nums[k]:
        print(-1)
    else:
        print(nums[k-1])

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240406110127244](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240406110127244.png)



### 27948: FBI树

http://cs101.openjudge.cn/practice/27948/



思路：读取01串，用函数判断其类型，并建树，若长度大于1则前半段为左子树，后半段为右子树，最后输出其后序遍历



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 11:10:18 2024

@author: 20311
"""
class Tree:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None

def check(s):
    a=0
    b=0
    for ss in s:
        if ss=='0':
            a+=1
        elif ss=='1':
            b+=1
        if a>0 and b>0:
            return 'F'
    if b==0:
        return 'B'
    elif a==0:
        return 'I'

def build(s):
    node=Tree(check(s))
    if len(s)<=1:
        return node
    s1=s[:len(s)//2]
    s2=s[len(s)//2:]
    node.left=build(s1)
    node.right=build(s2)
    return node

def traverse(node):
    if not node:
        return [node.value]
    
    ans=[]
    if node.left:
        ans+=traverse(node.left)
        
    if node.right:
        ans+=traverse(node.right)
    
    ans+=[node.value]
    return ans

n=input()
string=input().strip()
root=build(string)
ans=traverse(root)
print(''.join(ans))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240406113353176](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240406113353176.png)



### 27925: 小组队列

http://cs101.openjudge.cn/practice/27925/



思路：先分组，然后队列中按组储存顺序



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 11:36:26 2024

@author: 20311
"""

t=int(input())
d={}
for i in range(t):
    members=list(map(int,input().split()))
    for member in members:
        d[member]=i
queue=[]
while True:
    order=input()
    if order[:4]=='STOP':
        break
    elif order[:7]=='ENQUEUE':
        jg=False
        p=order.split()
        p=int(p[1])
        g=d[p]
        for i in range(len(queue)):
            if d[queue[i][0]]==g:
                queue[i].append(p)
                jg=True
                break
        if not jg:
            queue.append([p])
                
    else:
        ans=queue[0].pop(0)
        if len(queue[0])==0:
            queue.pop(0)
        print(ans)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240406115536367](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240406115536367.png)



### 27928: 遍历树

http://cs101.openjudge.cn/practice/27928/



思路：使用列表记录节点及子节点，然后从根开始遍历，遍历时对节点本身及子节点进行排序，逐个遍历，如果是子节点则递归，如果是自身则直接加入答案



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 17:05:14 2024

@author: 20311
"""
def traverse(node):
    if not d[node]:
        return [node]
    
    ans=[]
    order=[node]+d[node]
    order.sort()
    for o in order:
        if o==node:
            ans.append(node)
        else:
            ans+=traverse(o)
    return ans

d={}
nums={}
n=int(input())
for _ in range(n):
    data=input().strip()
    if ' ' in data:
        data=list(map(int,data.split()))
        d[data[0]]=data[1:]
        if data[0] not in nums:
            nums[data[0]]=True
        for da in data[1:]:
            nums[da]=False
    else:
        d[int(data)]=[]
        
for num in nums:
    if nums[num]:
        root=num
        
ans=traverse(root)
for a in ans:
    print(a)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240406172338065](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240406172338065.png)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

有

本次月考比起之前的题目还是比较简单的，但是第三题由于忽视了一些特殊情况卡了比较久，以后需要注意。同时还应当刷更多题目，提高写代码的熟练度，加快速度。



