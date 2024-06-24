# Assignment #F: All-Killed 满分

Updated 1844 GMT+8 May 20, 2024

2024 spring, Complied by ==狄晨阳 2300012138==



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

### 22485: 升空的焰火，从侧面看

http://cs101.openjudge.cn/practice/22485/



思路：由于节点的序号是随机分布的，所以需要先用一个列表记录所有节点，再根据输入将其连接起来，遍历时只需分层处理，将每层最右边一位输出即可



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:23:53 2024

@author: 20311
"""

class Node:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None
        
n=int(input())
nodes=[Node(i) for i in range(n+1)]
for i in range(1,n+1):
    x,y=map(int,input().split())
    a=nodes[i]
    if x!=-1:
        a.left=nodes[x]
    if y!=-1:
        a.right=nodes[y]
    nodes[i]=a

ans=[]
stack=[[nodes[1]]]
while stack:
    level=stack.pop(0)
    ans.append(level[-1].value)
    pre=[]
    for node in level:
        if node.left:
            pre.append(node.left)
        if node.right:
            pre.append(node.right)
    if pre:
        stack.append(pre)
        
print(' '.join(map(str,ans)))

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240523154028028](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240523154028028.png)



### 28203:【模板】单调栈

http://cs101.openjudge.cn/practice/28203/



思路：一道单调栈的模版题，参考题解学习了单调栈的写法



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:43:48 2024

@author: 20311
"""

n=int(input())
data=list(map(int,input().split()))
stack=[]
for i in range(n):
    while stack and data[stack[-1]]<data[i]:
        data[stack.pop()]=i+1
        
    stack.append(i)

while stack:
    data[stack[-1]]=0
    stack.pop()
    
print(*data)

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240523162648547](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240523162648547.png)



### 09202: 舰队、海域出击！

http://cs101.openjudge.cn/practice/09202/



思路：采用了拓扑排序的方式来进行深度优先搜索来判断有没有环



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Sat May 25 16:20:10 2024

@author: 20311
"""

from collections import defaultdict

def dfs(i):
    visited[i]=1
    for j in graph[i]:
        in_degree[j]-=1
        if in_degree[j]==0:
            dfs(j)

t=int(input())
for _ in range(t):
    n,m=map(int,input().split())
    graph=defaultdict(list)
    in_degree=[0]*(n+1)
    visited=[0]*(n+1)
    for _ in range(m):
        x,y=map(int,input().split())
        graph[x].append(y)
        in_degree[y]+=1
    for i in range(1,n+1):
        if in_degree[i]==0 and visited[i]!=1:
            dfs(i)
            
    if n!=visited.count(1):
        print('Yes')
    else:
        print('No')

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240525170559332](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240525170559332.png)



### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135/



思路：使用二分查找直接寻找答案



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:21:59 2024

@author: 20311
"""

n,m=map(int,input().split())
expense=[]
for _ in range(n):
    expense.append(int(input()))
    
left=max(expense)
right=sum(expense)

def check(mid):
    c=1
    s=0
    for i in expense:
        if s+i>mid:
            c+=1
            s=i
        else:
            s+=i
            
    if c>m:
        return True
    return False

ans=1
while left<right:
    mid=(left+right)//2
    if check(mid):
        left=mid+1
    
    else:
        ans=mid
        right=mid
        
print(ans)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240525174039965](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240525174039965.png)



### 07735: 道路

http://cs101.openjudge.cn/practice/07735/



思路：比较典型的dijkstra算法



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Sat May 25 19:07:02 2024

@author: 20311
"""
from heapq import heappop,heappush
from collections import defaultdict

def dij(roads):
    q=[(0,1,0)]
    while q:
        dis,fr,cost=heappop(q)
        if fr==n:
            return dis
        for new_dis,to,new_cost in roads[fr]:
            new_cost=cost+new_cost
            new_dis=dis+new_dis
            if new_cost<=k:
                heappush(q,(new_dis,to,new_cost))
    
    return -1

k=int(input())
n=int(input())
r=int(input())
roads=defaultdict(list)
for _ in range(r):
    fr,to,dis,cost=map(int,input().split())
    roads[fr].append((dis,to,cost))
print(dij(roads))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240525194050964](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240525194050964.png)



### 01182: 食物链

http://cs101.openjudge.cn/practice/01182/



思路：并查集但是比较抽象，还需要加深理解



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Sat May 25 20:42:26 2024

@author: 20311
"""

def find(x):
    if p[x]==x:
        return x
    else:
        p[x]=find(p[x])
        return p[x]

n,k=map(int,input().split())
p=[x for x in range(3*n+1)]
ans=0
for _ in range(k):
    d,i,j=map(int,input().split())
    if i>n or j>n:
        ans+=1
        continue
    
    if d==1:
        if find(i+n)==find(j) or find(j+n)==find(i):
            ans+=1 
            continue
        
        p[find(i)]=find(j)
        p[find(i+n)]=find(j+n)
        p[find(i+2*n)]=find(j+2*n)
    else:
        if find(i)==find(j) or find(j+n)==find(i):
            ans+=1
            continue
        
        p[find(i+n)]=find(j)
        p[find(j+2*n)]=find(i)
        p[find(i+2*n)]=find(j+n)
        
print(ans)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240526002030544](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240526002030544.png)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

有

本次的题目涉及的算法比较全面，可以说基本上很多都复习了一遍，感觉还是不大熟练，需要多加练习



