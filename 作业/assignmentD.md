# Assignment #D: May月考

Updated 1654 GMT+8 May 8, 2024

2024 spring, Complied by ==狄晨阳 生命科学学院==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：Window11

Python编程环境：Spyder IDE 5.4.3

C/C++编程环境：无

## 1. 题目

### 02808: 校门外的树

http://cs101.openjudge.cn/practice/02808/



思路：用一个列表来处理数据



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Tue May 21 19:25:06 2024

@author: 20311
"""

l,m=map(int,input().split())
cut=[1]*(l+1)
for _ in range(m):
    a,b=map(int,input().split())
    for i in range(max(a,0),min(b+1,l+1)):
        cut[i]=0
print(cut.count(1))

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240521193503207](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240521193503207.png)



### 20449: 是否被5整除

http://cs101.openjudge.cn/practice/20449/



思路：按题目要求处理即可



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Tue May 21 19:36:02 2024

@author: 20311
"""

a=input()
ans=''
for i in range(1,len(a)+1):
    b=int(a[:i],2)
    if b%5==0:
        ans+='1'
    else:
        ans+='0'
print(ans)

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240521194013162](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240521194013162.png)



### 01258: Agri-Net

http://cs101.openjudge.cn/practice/01258/



思路：使用prim算法来找出最短连接长度



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Tue May 21 19:45:01 2024

@author: 20311
"""

from heapq import heappop,heappush
while True:
    try:
        n=int(input())
        matrix=[]
        for i in range(n):
            matrix.append(list(map(int,input().split())))
        d=[100000]*n
        visited=set()
        q=[]
        l=0
        d[0]=0
        heappush(q,(d[0],0))
        while q:
            dis,p=heappop(q)
            if p in visited:
                continue
            visited.add(p)
            l+=d[p]
            for i in range(n):
                if d[i]>matrix[p][i]:
                    d[i]=matrix[p][i]
                    heappush(q,(d[i],i))
        print(l)
    except EOFError:
        break

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240521200344773](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240521200344773.png)



### 27635: 判断无向图是否连通有无回路(同23163)

http://cs101.openjudge.cn/practice/27635/



思路：在进行bfs的同时记录每一次的父节点，只要通向的新节点是已经经过的且不为父节点即判断为有回路，bfs结束后如果经过了所有节点即为全部连接



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Tue May 21 20:07:02 2024

@author: 20311
"""

n,m=map(int,input().split())
d={x:[] for x in range(n)}
d[-1]=[]
for _ in range(m):
    a,b=map(int,input().split())
    d[a].append(b)
    d[b].append(a)
jg1=False
jg2=False
q=[[-1,[a]]]
v={a}
while q:
    b=q.pop(0)
    for i in range(len(b[1])):
        c=[]
        x=b[1][i]
        for p in d[x]:
            if p not in v:
                v.add(p)
                c.append(p)
            elif p!=b[0]:
                jg2=True
        if c:
            q.append([x,c])
if len(v)==n:
    jg1=True
    
if jg1:
    print('connected:yes')
else:
    print('connected:no')
    
if jg2:
    print('loop:yes')
else:
    print('loop:no')

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240521205402893](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240521205402893.png)





### 27947: 动态中位数

http://cs101.openjudge.cn/practice/27947/



思路：使用两个堆来维护中位数使其永远在大根堆的堆顶



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:00:11 2024

@author: 20311
"""
from heapq import heappop,heappush
def process(data):
    m1=[]
    m2=[]
    ans=[]
    for i in range(len(data)):
        num=data[i]
        if not m1 or num<=-m1[0]:
            heappush(m1,-num)
        else:
            heappush(m2,num)
            
        if len(m1)-len(m2)>1:
            heappush(m2,-heappop(m1))
        elif len(m2)>len(m1):
            heappush(m1,-heappop(m2))
            
        if i%2==0:
            ans.append(-m1[0])
    
    return ans

n=int(input())
for _ in range(n):
    data=list(map(int,input().split()))
    ans=process(data)
    print(len(ans))
    print(' '.join(map(str,ans)))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240521211249091](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240521211249091.png)



### 28190: 奶牛排队

http://cs101.openjudge.cn/practice/28190/



思路：没有想清楚该如何用数学语言来描述此题并简化时间复杂度，查看题解了解到应该使用两个单调栈来处理



代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Tue May 21 21:28:34 2024

@author: 20311
"""

N = int(input())
heights = [int(input()) for _ in range(N)]

left_bound = [-1] * N
right_bound = [N] * N

stack = []
for i in range(N):
    while stack and heights[stack[-1]] < heights[i]:
        stack.pop()

    if stack:
        left_bound[i] = stack[-1]

    stack.append(i)

stack = []
for i in range(N-1, -1, -1):
    while stack and heights[stack[-1]] > heights[i]:
        stack.pop()

    if stack:
        right_bound[i] = stack[-1]

    stack.append(i)

ans = 0


for i in range(N):
    for j in range(left_bound[i] + 1, i):
        if right_bound[j] > i:
            ans = max(ans, i - j + 1)
            break
print(ans)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240521213120175](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240521213120175.png)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

有

本次月考除了最后一题还是比较好想的，但是仍然需要提高熟练度，以及对题意的理解与翻译能力



