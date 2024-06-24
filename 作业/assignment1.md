# Assignment #1: 拉齐大家Python水平

Updated 0940 GMT+8 Feb 19, 2024

2024 spring, Complied by ==狄晨阳 生命科学学院==



**说明：**

1）数算课程的先修课是计概，由于计概学习中可能使用了不同的编程语言，而数算课程要求Python语言，因此第一周作业练习Python编程。如果有同学坚持使用C/C++，也可以，但是建议也要会Python语言。

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

### 20742: 泰波拿契數

http://cs101.openjudge.cn/practice/20742/



思路：打表输出所有项再从中检索所需的



##### 代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 21:16:57 2024

@author: 20311
"""

l=[0,1,1]
for n in range(3,31):
    l.append(l[n-1]+l[n-2]+l[n-3])
n=int(input())
print(l[n])

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240220214252699](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240220214252699.png)



### 58A. Chat room

greedy/strings, 1000, http://codeforces.com/problemset/problem/58/A



思路：一遍扫过去，用表的五个位置来实现循序渐进的判定存在性



##### 代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 21:22:51 2024

@author: 20311
"""

s=input()
d=[-1,-1,-1,-1,-1]
for i in range(len(s)):
    if s[i]=='h' and d[0]==-1:
        d[0]=i
    elif s[i]=='e' and d[1]<d[0]:
        d[1]=i
    elif s[i]=='l' and d[2]<d[1]:
        d[2]=i
    elif s[i]=='l' and d[3]<d[2]:
        d[3]=i
    elif s[i]=='o' and d[4]<d[3]:
        d[4]=i
if d[4]!=-1:
    print('YES')
else:
    print('NO')

```



代码运行截图 ==（至少包含有"Accepted"）==

![image-20240220214449880](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240220214449880.png)



### 118A. String Task

implementation/strings, 1000, http://codeforces.com/problemset/problem/118/A



思路：用一个集合储存需要删除的字母，输入直接转换为小写保存，不用删的前面加点直接写在输出里



##### 代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 21:46:56 2024
 
@author: 20311
"""
 
s=list(input().lower())
d={'a','e','i','o','u','y'}
r=''
for a in s:
    if a in d:
        pass
    else:
        r+='.'
        r+=a
print(r)

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240220215901433](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240220215901433.png)



### 22359: Goldbach Conjecture

http://cs101.openjudge.cn/practice/22359/



思路：通过函数来判断是不是质数，用穷举找答案，如果有则输出并结束程序



##### 代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:25:11 2024

@author: 20311
"""

def jg(a):
    for i in range(2,int(a**0.5)+1):
        if a%i==0:
            return False
    return True

s=int(input())
for a in range(2,s//2+1):
    if jg(a) and jg(s-a):
        print(a,s-a)
        break

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240221183111455](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240221183111455.png)



### 23563: 多项式时间复杂度

http://cs101.openjudge.cn/practice/23563/



思路：先用加号分割每个项，然后分别处理，对系数不为0的项用"^"分割，去最后一项以提取出其指数，储存最大的指数并以所需形式输出



##### 代码

```python
#  -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:33:51 2024

@author: 20311
"""

a=input().split('+')
m=0
for s in a:
    if s[0]!='0':
        b=int(s.split('^')[-1])
        m=max(m,b)
print('n^{}'.format(m))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240221184017691](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240221184017691.png)



### 24684: 直播计票

http://cs101.openjudge.cn/practice/24684/



思路：一个个读取数据，用字典储存票数并统计实时最大值，然后检索所有最大值的序号并输出



##### 代码

```python
# # -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 18:43:24 2024

@author: 20311
"""

d={}
a=list(map(int,input().split()))
m=0
for b in a:
    if b not in d:
        d[b]=1
    else:
        d[b]+=1
    m=max(m,d[b])
r=[]
for i in d:
    if d[i]==m:
        r.append(i)
r.sort()
print(' '.join(map(str,r)))

```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![image-20240221184856572](C:\Users\20311\AppData\Roaming\Typora\typora-user-images\image-20240221184856572.png)



## 2. 学习总结和收获

==如果作业题目简单，有否额外练习题目，比如：OJ“数算pre每日选做”、CF、LeetCode、洛谷等网站题目。==

有

经过了一个学期的学习，可以看出现在写代码比以前的好了很多，很有成就感，还要继续努力



