# Assignment #2: 编程练习

Updated 0953 GMT+8 Feb 24, 2024

2024 spring, Complied by ==狄晨阳 生命科学学院==



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:
- Learn about Time and Space complexities
- Learn the basics of individual Data Structures
- Learn the basics of Algorithms
- Practice Problems on DSA

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知3月1日导入选课名单后启用。**作业写好后，保留在自己手中，待3月1日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：Windows11

Python编程环境：Spyder IDE 5.4.3

C/C++编程环境：无



## 1. 题目

### 27653: Fraction类

http://cs101.openjudge.cn/2024sp_routine/27653/



思路：用通式求计算后的分子分母，在穷举找最大公因数求出最简形式



##### 代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:54:52 2024

@author: 20311
"""

a,b,c,d=map(int,input().split())
fm=b*d
fz=a*d+b*c
ys=1
for i in range(2,min(fm,fz)+1):
    if fm%i==0 and fz%i==0:
        ys=i
print("{}/{}".format(fz//ys, fm//ys))

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240227170137908](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240227170137908.png)



### 04110: 圣诞老人的礼物-Santa Clau’s Gifts

greedy/dp, http://cs101.openjudge.cn/practice/04110



思路：用平均价格排序，从大到小装入直到装满



##### 代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:02:46 2024

@author: 20311
"""

n,m=map(int,input().split())
c=[]
for _ in range(n):
    c.append(list(map(int,input().split())))
c.sort(key=lambda x:x[0]/x[1],reverse=True)
i=0
ans=0
while m>0 and i<n:
    if m>=c[i][1]:
        ans+=c[i][0]
        m-=c[i][1]
        i+=1
    else:
        ans+=c[i][0]*m/c[i][1]
        break
print(f'{ans:.1f}')

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240227171853752](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240227171853752.png)



### 18182: 打怪兽

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/



思路：用字典储存每个时刻的数据，将键转到字典中排序，然后按先后顺序，对每个时刻的技能伤害排序并减去，最后输出



##### 代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:20:54 2024

@author: 20311
"""

x=int(input())
for _ in range(x):
    n,m,b=map(int,input().split())
    d={}
    for z in range(n):
        ti,xi=map(int,input().split())
        if ti in d:
            d[ti].append(xi)
        else:
            d[ti]=[xi]
    t=list(d.keys())
    t.sort()
    for i in t:
        d[i].sort(reverse=True)
        b-=sum(d[i][:m])
        if b<=0:
            break
    if b<=0:
        print(i)
    else:
        print('alive')

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240227173443372](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240227173443372.png)



### 230B. T-primes

binary search/implementation/math/number theory, 1300, http://codeforces.com/problemset/problem/230/B



思路：使用了埃氏筛来筛出10**6以下的质数，判断每一个数据是否是质数的平方



##### 代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 18:31:45 2024
 
@author: 20311
"""
from math import ceil
p=[True]*(10**6+1)
pp=set()
p[0]=p[1]=False
for i in range(4,10**6+1,2):
    p[i]=False
for i in range(3,(10**6+1)//3+2,2):
    if p[i]:
        for k in range(2,ceil((10**6+1)/i)):
            p[i*k]=False
for i in range(10**6+1):
    if p[i]:
        pp.add(i**2)
input()
data=list(map(int,input().split()))
for d in data:
    if d in pp:
        print('YES')
    else:
        print('NO')

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240227185619665](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240227185619665.png)



### 1364A. XXXXX

brute force/data structures/number theory/two pointers, 1200, https://codeforces.com/problemset/problem/1364/A



思路：如果总和不满足，只要从左右两边分别求和，直到和不是x的倍数，然后取最大长度，若不存在就输出-1



##### 代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:00:05 2024
 
@author: 20311
"""
 
t=int(input())
for _ in range(t):
    n,x=map(int,input().split())
    data=list(map(int,input().split()))
    data.append(0)
    s=sum(data)
    if s%x!=0:
        print(n)
    else:
        l=0
        for i in range(n):
            l+=data[i]
            if l%x!=0:
                break
        r=0
        for j in range(n-1,-1,-1):
            r+=data[j]
            if r%x!=0:
                break
        if i==n-1 and j!=0:
            print(j)
        elif i!=n-1 and j==0:
            print(n-i-1)
        elif i!=n-1 and j!=0:
            print(max(j,n-i-1))
        else:
            print(-1)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240227194523734](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240227194523734.png)



### 18176: 2050年成绩计算

http://cs101.openjudge.cn/practice/18176/



思路：对选课数计数，判断每个成绩是不是t-prime然后求得答案



##### 代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:47:11 2024

@author: 20311
"""
from math import ceil
p=[True]*(10**4+1)
pp=set()
p[0]=p[1]=False
for i in range(4,10**4+1,2):
    p[i]=False
for i in range(3,(10**4+1)//3+2,2):
    if p[i]:
        for k in range(2,ceil((10**4+1)/i)):
            p[i*k]=False
for i in range(10**4+1):
    if p[i]:
        pp.add(i**2)
m,n=list(map(int,input().split()))
for _ in range(m):
    cj=list(map(int,input().split()))
    ans=0
    i=0
    for c in cj:
        i+=1
        if c in pp:
            ans+=c
    if ans!=0:
        avg=ans/i
        print(f'{avg:.2f}')
    else:
        print(0)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240227195612536](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240227195612536.png)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==

有

虽然代码更合理了，但是对于暴力、贪心一类的题目还是想得太简单导致超时，还是要多通过思维时间来换取更短的运行时间



