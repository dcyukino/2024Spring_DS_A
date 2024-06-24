### 前言

由于时间问题并未全程自己做cheatsheet，此cheatsheet是在同学的基础上修改添加而来

### 经典题

前中得后序

```python
def postorder(preorder,inorder):
    if not preorder:
        return ''
    root=preorder[0]
    idx=inorder.index(root)
    left=postorder(preorder[1:idx+1],inorder[:idx])
    right=postorder(preorder[idx+1:],inorder[idx+1:])
    return left+right+root
```

中后得前序

```python
def preorder(inorder,postorder):
    if not inorder:
        return ''
    root=postorder[-1]
    idx=inorder.index(root)
    left=preorder(inorder[:idx],postorder[:idx])
    right=preorder(inorder[idx+1:],postorder[idx:-1])
    return root+left+right
```

层次遍历

```python
from collections import deque
def levelorder(root):
    if not root:
        return ""
    q=deque([root])  
    res=""
    while q:
        node=q.popleft()  
        res+=node.val  
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return res
```

括号嵌套表达式

```python
def parse(s):
    node=Node(s[0])
    if len(s)==1:
        return node
    s=s[2:-1]; t=0; last=-1
    for i in range(len(s)):
        if s[i]=='(': t+=1
        elif s[i]==')': t-=1
        elif s[i]==',' and t==0:
            node.children.append(parse(s[last+1:i]))
            last=i
    node.children.append(parse(s[last+1:]))
    return node
```

构建二叉搜索树

```python
def insert(root,num):
    if not root:
        return Node(num)
    if num<root.val:
        root.left=insert(root.left,num)
    else:
        root.right=insert(root.right,num)
    return root
```

并查集

```python
class UnionFind:
    def __init__(self,n):
        self.p=list(range(n))
        self.h=[0]*n
    def find(self,x):
        if self.p[x]!=x:
            self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self,x,y):
        rootx=self.find(x)
        rooty=self.find(y)
        if rootx!=rooty:
            if self.h[rootx]<self.h[rooty]:
                self.p[rootx]=rooty
            elif self.h[rootx]>self.h[rooty]:
                self.p[rooty]=rootx
            else:
                self.p[rooty]=rootx
                self.h[rootx]+=1
```

构建字典树

```python
def insert(root,num):
    node=root
    for digit in num:
        if digit not in node.children:
            node.children[digit]=TrieNode()
        node=node.children[digit]
        node.cnt+=1
```

bfs

```python
from collections import deque
def bfs(graph, start_node):
    queue = deque([start_node])
    visited = set()
    visited.add(start_node)
    while queue:
        current_node = queue.popleft()
        for neighbor in graph[current_node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

dijkstra

```python
# 1.使用vis集合
def dijkstra(start,end):
    heap=[(0,start,[start])]
    vis=set()
    while heap:
        (cost,u,path)=heappop(heap)
        if u in vis: continue
        vis.add(u)
        if u==end: return (cost,path)
        for v in graph[u]:
            if v not in vis:
                heappush(heap,(cost+graph[u][v],v,path+[v]))
# 2.使用dist数组
import heapq
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    return distances
```

拓扑排序

```python
from collections import deque
def topo_sort(graph):
    in_degree={u:0 for u in graph}
    for u in graph:
        for v in graph[u]:
            in_degree[v]+=1
    q=deque([u for u in in_degree if in_degree[u]==0])
    topo_order=[]
    while q:
        u=q.popleft()
        topo_order.append(u)
        for v in graph[u]:
            in_degree[v]-=1
            if in_degree[v]==0:
                q.append(v)
    if len(topo_order)!=len(graph):
        return []  
    return topo_order
```

Bfs最小路径

```python
from collections import deque

def bfs(m, n, grid):
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    visited = [[False] * n for _ in range(m)]
    start = (0, 0)
    queue = deque([(start, 0)])  # 起点和步数入队
    visited[start[0]][start[1]] = True

    while queue:
        current, steps = queue.popleft()
        x, y = current

        if grid[x][y] == 1:  # 到达藏宝点
            return steps

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] != 2 and not visited[nx][ny]:
                visited[nx][ny] = True
                queue.append(((nx, ny), steps + 1))

    return -1  # 无法到达藏宝点

m, n = map(int, input().split())
grid = []

for _ in range(m):
    row = list(map(int, input().split()))
    grid.append(row)

result = bfs(m, n, grid)

if result == -1:
    print("NO")
else:
    print(result)
    
```

dp最大上升子序列

```python
input()
b = [int(x) for x in input().split()]

n = len(b)
dp = [0]*n

for i in range(n):
    dp[i] = b[i]
    for j in range(i):
        if b[j]<b[i]:
            dp[i] = max(dp[j]+b[i], dp[i])
    
print(max(dp))
```

堆

```python
import heapq
x = [1,2,3,5,7]

heapq.heapify(x)
###将列表转换为堆。

heapq.heappushpop(heap, item)
##将 item 放入堆中，然后弹出并返回 heap 的最小元素。该组合操作比先调用 heappush() 再调用 heappop() 运行起来更有效率

heapq.heapreplace(heap, item)
##弹出并返回最小的元素，并且添加一个新元素item

heapq.heappop(heap,item)
heapq.heappush(heap,item)
```

单调栈

```python
n = int(input())
lst = list(int(i) for i in input().split())
pos = [0 for i in range(n)] ##初始化位置
judge = lst[0]
stack = []
for i in range(n-1,-1,-1):
    while stack and lst[stack[-1]] <= lst[i]:
        stack.pop()
    if stack:
        pos[i] = stack[-1] + 1
    stack.append(i)
print(*pos)
```

插入排序

```python
def insertion_sort(arr):							
    for i in range(1, len(arr)):
        j = i										
        while arr[j - 1] > arr[j] and j > 0:		
            arr[j - 1], arr[j] = arr[j], arr[j - 1]
arr = [2, 6, 5, 1, 3, 4]
insertion_sort(arr)
print(arr)

# [1, 2, 3, 4, 5, 6]
```

冒泡排序

```python
def bubbleSort(arr):
    n = len(arr)
    for i in range(n):	# (*)
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if (swapped == False):
            break

if __name__ == "__main__":
    arr = [64, 34, 25, 12, 22, 11, 90]
    bubbleSort(arr)
    print(' '.join(map(str, arr)))
```

选择排序

```python
A = [64, 25, 12, 22, 11]
# 一位一位往下找，确保每一趟后，该位及之前的元素有序。
for i in range(len(A)):
    min_idx = i
    for j in range(i + 1, len(A)):
        if A[min_idx] > A[j]:
            min_idx = j
    A[i], A[min_idx] = A[min_idx], A[i]
    
print(' '.join(map(str, A)))
# Output: 11 12 22 25 64 
```

快速排序

```python
def quicksort(arr, left, right):
    # 函数的功能就是把数组从left到right排成顺序。
    if left < right:
        partition_pos = partition(arr, left, right)
        quicksort(arr, left, partition_pos - 1)
        quicksort(arr, partition_pos + 1, right)

def partition(arr, left, right):
    # 函数的功能是：把数组从left到right依据pivot分成两部分，其中pivot左边小于pivot,右半部分不小于pivot.
    i = left
    j = right - 1
    pivot = arr[right]
    while i <= j:
        # 筛选不合适的arr[i]，即在pivot左边且大于等于pivot
        while i <= right and arr[i] < pivot:
            i += 1
        # 筛选不合适的arr[j],即在pivot右边且小于pivot
        while j >= left and arr[j] >= pivot:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
    if arr[i] > pivot:
        arr[i], arr[right] = arr[right], arr[i]
    return i


arr = [22, 11, 88, 66, 55, 77, 33, 44]
quicksort(arr, 0, len(arr) - 1)
print(arr)

# [11, 22, 33, 44, 55, 66, 77, 88]
```

### 可能用到的库

### deque

`deque`（双端队列）是一个从头部和尾部都能快速增删元素的容器。这种数据结构非常适合用于需要快速添加和弹出元素的场景，如队列和栈。

1. 添加元素

- **`append(x)`**：在右端添加一个元素 `x`。时间复杂度为 O(1)。
- **`appendleft(x)`**：在左端添加一个元素 `x`。时间复杂度为 O(1)。

2. 移除元素

- **`pop()`**：移除并返回右端的元素。如果没有元素，将引发 `IndexError`。时间复杂度为 O(1)。
- **`popleft()`**：移除并返回左端的元素。如果没有元素，将引发 `IndexError`。时间复杂度为 O(1)。

3. 扩展

- **`extend(iterable)`**：在右端依次添加 `iterable` 中的元素。整体操作的时间复杂度为 O(k)，其中 `k` 是 `iterable` 的长度。
- **`extendleft(iterable)`**：在左端依次添加 `iterable` 中的元素。注意，添加的顺序会是 `iterable` 元素的逆序。整体操作的时间复杂度为 O(k)，其中 `k` 是 `iterable` 的长度。

4. 其他操作

- **`rotate(n=1)`**：向右旋转队列 `n` 步。如果 `n` 是负数，则向左旋转。这个操作的时间复杂度为 O(k)，其中 `k` 是 `n` 的绝对值，但实际上因为只涉及到指针移动，所以非常快。
- **`clear()`**：移除所有的元素，使其长度为 0。时间复杂度为 O(n)，其中 `n` 是 `deque` 中元素的数量。
- **`remove(value)`**：移除找到的第一个值为 `value` 的元素。这个操作在最坏情况下的时间复杂度为 O(n)，因为可能需要遍历整个 `deque`。

5. 访问元素

- 对于 `deque`，虽然可以通过索引访问，如 `d[0]` 或 `d[-1]`，但这不是 `deque` 设计的主要用途，且访问中间元素的时间复杂度为 O(n)。因此，如果你需要频繁地从随机位置访问数据，`deque` 可能不是最佳选择。

```python
from collections import deque

# 初始化deque
d = deque([1, 2, 3])

# 添加元素
d.append(4)  # deque变为[1, 2, 3, 4]
d.appendleft(0)  # deque变为[0, 1, 2, 3, 4]

# 移除元素
d.pop()  # 返回 4, deque变为[0, 1, 2, 3]
d.popleft()  # 返回 0, deque变为[1, 2, 3]

# 扩展
d.extend([4, 5])  # deque变为[1, 2, 3, 4, 5]
d.extendleft([0])  # deque变为[0, 1, 2, 3, 4, 5]

# 旋转
d.rotate(1)  # deque变为[5, 0, 1, 2, 3, 4]
d.rotate(-2)  # deque变为[1, 2, 3, 4, 5, 0]

# 清空
d.clear()  # deque变为空
```

### Counter, defaultdict, namedtuple, OrderedDict

1. `Counter`

`Counter` 是一个用于计数可哈希对象的字典子类。它是一个集合，其中元素的存储形式为字典键值对，键是元素，值是元素计数。

```python
from collections import Counter

# 创建 Counter 对象
cnt = Counter(['red', 'blue', 'red', 'green', 'blue', 'blue'])

# 访问计数
print(cnt['blue'])    # 输出: 3
print(cnt['red'])     # 输出: 2

# 更新计数
cnt.update(['blue', 'red', 'blue'])
print(cnt['blue'])    # 输出: 5

# 计数的常见方法
print(cnt.most_common(2))  # 输出 [('blue', 5), ('red', 3)]
```

2. `defaultdict`

`defaultdict` 是另一种字典子类，它提供了一个默认值，用于字典所尝试访问的键不存在时返回。

```python
from collections import defaultdict

# 使用 lambda 来指定默认值为 0
d = defaultdict(lambda: 0)

d['key1'] = 5
print(d['key1'])  # 输出: 5
print(d['key2'])  # 输出: 0，因为 key2 不存在，返回默认值 0
```

3. `namedtuple`

`namedtuple` 生成可以使用名字来访问元素内容的元组子类。

```python
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(11, y=22)

print(p.x + p.y)  # 输出: 33
print(p[0] + p[1])  # 输出: 33  # 还可以像普通元组那样用索引访问
```

4. `OrderedDict`

`OrderedDict` 是一个字典子类，它保持了元素被添加的顺序，这在某些情况下非常有用。

```python
from collections import OrderedDict

od = OrderedDict()
od['z'] = 1
od['y'] = 2
od['x'] = 3

for key in od:
    print(key, od[key])
# 输出:
# z 1
# y 2
# x 3
```

## permutations

在 Python 中，`permutations` 是 `itertools` 模块中的一个非常有用的函数，用于生成输入可迭代对象的所有可能排列。排列是将一组元素组合成一定顺序的所有可能方式。例如，集合 [1, 2, 3] 的全排列包括 [1, 2, 3]、[1, 3, 2]、[2, 1, 3] 等。

使用 `itertools.permutations`

`itertools.permutations(iterable, r=None)` 函数接收两个参数：

- `iterable`：要排列的数据集。
- `r`：可选参数，指定生成排列的长度。如果 `r` 未指定，则默认值等于 `iterable` 的长度，即生成全排列。

返回值是一个迭代器，生成元组，每个元组是一个可能的排列。

示例代码

下面是使用 `itertools.permutations` 的一些示例：

1. 生成全排列

```python
import itertools

data = [1, 2, 3]
permutations_all = list(itertools.permutations(data))

# 输出所有排列
for perm in permutations_all:
    print(perm)
```

输出：

```python
(1, 2, 3)
(1, 3, 2)
(2, 1, 3)
(2, 3, 1)
(3, 1, 2)
(3, 2, 1)
```

2. 生成长度为 `r` 的排列

如果你只想生成一部分元素的排列，可以设置 `r` 的值。

```python
import itertools

data = [1, 2, 3, 4]
permutations_r = list(itertools.permutations(data, 2))

# 输出长度为2的排列
for perm in permutations_r:
    print(perm)
```

输出：

```python
(1, 2)
(1, 3)
(1, 4)
(2, 1)
(2, 3)
(2, 4)
(3, 1)
(3, 2)
(3, 4)
(4, 1)
(4, 2)
(4, 3)
```

注意事项

- `itertools.permutations` 生成的排列是 **不重复的**，即使输入的元素中有重复，输出的每个排列仍然是唯一的。
- 生成的排列是按照字典序排列的，基于输入 `iterable` 的顺序。
- 由于排列的数量非常快地随着 `n`（元素总数）和 `r`（排列的长度）的增加而增加，生成非常大的排列集可能会消耗大量的内存和计算资源。例如，10个元素的全排列总共有 10! (即 3,628,800) 种可能，这在实际应用中可能是不切实际的。

使用 `itertools.permutations` 可以有效地处理排列问题，是解决许多算法问题的有力工具。

## heapq

`heapq` 模块是 Python 的标准库之一，提供了基于堆的优先队列算法的实现。堆是一种特殊的完全二叉树，满足父节点的值总是小于或等于其子节点的值（在最小堆的情况下）。这个属性使堆成为实现优先队列的理想数据结构。

基本操作

`heapq` 模块提供了一系列函数来管理堆，但它只提供了“最小堆”的实现。以下是一些主要功能及其用法：

1. `heapify(x)`

- **用途**：将列表 `x` 原地转换为堆。

- 示例

  ```
  import heapq
  data = [3, 1, 4, 1, 5, 9, 2, 6, 5]
  heapq.heapify(data)
  print(data)  # 输出将是堆，但可能不是完全排序的
  ```

2. `heappush(heap, item)`

- **用途**：将 `item` 加入到堆 `heap` 中，并保持堆的不变性。

- 示例

  ```
  heap = []
  heapq.heappush(heap, 3)
  heapq.heappush(heap, 1)
  heapq.heappush(heap, 4)
  print(heap)  # 输出最小元素总是在索引0
  ```

3. `heappop(heap)`

- **用途**：弹出并返回 `heap` 中最小的元素，保持堆的不变性。

- 示例

  ```
  print(heapq.heappop(heap))  # 返回1
  print(heap)  # 剩余的堆
  ```

4. `heapreplace(heap, item)`

- **用途**：弹出堆中最小的元素，并将新的 `item` 插入堆中，效率高于先 `heappop()` 后 `heappush()`。

- 示例

  ```
  heapq.heapreplace(heap, 7)
  print(heap)
  ```

5. `heappushpop(heap, item)`

- **用途**：先将 `item` 压入堆中，然后弹出并返回堆中最小的元素。

- 示例

  ```
  result = heapq.heappushpop(heap, 0)
  print(result)  # 输出0
  print(heap)  # 剩余的堆
  ```

6. `nlargest(n, iterable, key=None)` 和 `nsmallest(n, iterable, key=None)`

- **用途**：从 `iterable` 数据中找出最大的或最小的 `n` 个元素。

- 示例

  ```
  data = [3, 1, 4, 1, 5, 9, 2, 6, 5]
  print(heapq.nlargest(3, data))  # 输出[9, 6, 5]
  print(heapq.nsmallest(3, data))  # 输出[1, 1, 2]
  ```

应用场景

`heapq` 通常用于需要快速访问最小（或最大）元素的场景，但不需要对整个列表进行完全排序。它广泛应用于数据处理、实时计算、优先级调度等领域。例如，任务调度、Dijkstra 最短路径算法、Huffman 编码树生成等都会用到堆结构。

注意事项

- 如需实现最大堆功能，可以通过对元素取反来实现。将所有元素取负后使用 `heapq`，然后再取负回来即可。
- 堆操作的时间复杂度一般为 O(log n)，适合处理大数据集。
- `heapq` 只能保证列表中的第一个元素是最小的，其他元素的排序并不严格。

## queue

Python 的 `queue` 模块提供了多种队列类型，主要用于线程间的通信和数据共享。这些队列都是线程安全的，设计用来在生产者和消费者线程之间进行数据交换。除了已经提到的 `LifoQueue` 之外，`queue` 模块还提供了以下几种有用的队列类型：

1. `Queue`

这是标准的先进先出（FIFO）队列。元素从队列的一端添加，并从另一端被移除。这种类型的队列特别适用于任务调度，保证了任务被处理的顺序。

- **`put(item, block=True, timeout=None)`**：将 `item` 放入队列中。如果可选参数 `block` 设为 `True`，并且 `timeout` 是一个正数，则在超时前会阻塞等待可用的槽位。
- **`get(block=True, timeout=None)`**：从队列中移除并返回一个元素。如果可选参数 `block` 设为 `True`，并且 `timeout` 是一个正数，则在超时前会阻塞等待元素。
- **`empty()`**：判断队列是否为空。
- **`full()`**：判断队列是否已满。
- **`qsize()`**：返回队列中的元素数量。注意，这个大小只是近似值，因为在返回值和队列实际状态间可能存在时间差。

2. `PriorityQueue`

基于优先级的队列，队列中的每个元素都有一个优先级，优先级最低的元素（注意是最“低”）最先被移除。这是通过将元素存储为 `(priority_number, data)` 对来实现的。

- 优先级可以是任何可排序的类型，通常是数字，其中较小的值具有较高的优先级。

3. `SimpleQueue`

在 Python 3.7 及以后版本中引入了 `SimpleQueue`，它是一个简单的先进先出队列，没有大小限制，不像 `Queue`，它没有任务跟踪或其他复杂的功能，通常性能更好。

- **`put(item)`**：将 `item` 放入队列。
- **`get()`**：从队列中移除并返回一个元素。
- **`empty()`**：判断队列是否为空。

4.`LifoQueue` 

在 Python 中，LIFO（后进先出）队列可以通过标准库中的 `queue` 模块实现，其中 `LifoQueue` 类提供了一个基于 LIFO 原则的队列实现。LIFO 队列通常被称为堆栈（stack），因为它遵循“后进先出”的原则，即最后一个添加到队列中的元素将是第一个被移除的元素。

`LifoQueue` 提供了以下几个主要的方法：

- **`put(item)`**: 将 `item` 元素放入队列中。
- **`get()`**: 从队列中移除并返回最顶端的元素。
- **`empty()`**: 检查队列是否为空。
- **`full()`**: 检查队列是否已满。
- **`qsize()`**: 返回队列中的元素数量。

示例代码

下面是如何使用 `queue.LifoQueue` 的一个简单示例：

```
import queue

# 创建一个 LIFO 队列
lifo_queue = queue.LifoQueue()

# 添加元素
lifo_queue.put('a')
lifo_queue.put('b')
lifo_queue.put('c')

# 依次取出元素
print(lifo_queue.get())  # 输出 'c'
print(lifo_queue.get())  # 输出 'b'
print(lifo_queue.get())  # 输出 'a'
```

注意事项

- `LifoQueue` 是线程安全的，这意味着它可以安全地用于多线程环境。
- 如果 `LifoQueue` 初始化时指定了最大容量，`put()` 方法在队列满时默认会阻塞，直到队列中有空闲位置。如果需要，可以用 `put_nowait()` 方法来避免阻塞，但如果队列满了，这会抛出 `queue.Full` 异常。
- 类似地，`get()` 方法在队列为空时会阻塞，直到队列中有元素可以取出。`get_nowait()` 方法也可以用来避免阻塞，但如果队列空了，会抛出 `queue.Empty` 异常。

示例代码

下面是一个使用 `PriorityQueue` 的例子：

```
import queue

# 创建一个优先级队列
pq = queue.PriorityQueue()

# 添加元素及其优先级
pq.put((3, 'Low priority'))
pq.put((1, 'High priority'))
pq.put((2, 'Medium priority'))

# 依次取出元素
while not pq.empty():
    print(pq.get()[1])  # 输出元素的数据部分
```

使用场景

- **`Queue`**: 适用于任务调度，如在多线程下载文件时管理下载任务。
- **`LifoQueue`**: 适用于需要后进先出逻辑的场景，比如回溯算法。
- **`PriorityQueue`**: 用于需要处理优先级任务的场景，如操作系统的任务调度。
- **`SimpleQueue`**: 适用于需要快速操作且不需要额外功能的场景，比如简单的数据传递任务。

这些队列因其线程安全的特性，特别适合用于多线程程序中，以确保数据的一致性和完整性。

### 1. 大小写转换

```python
text: str
text.upper() # 变全大写
text.lower() # 变全小写
text.capitalize() # 首字母大写
text.title() # 单个字母大写
text.swapcase() # 大小写转换
s[idx].isdigit() # 判断是否为整
s.isnumeric() # 判断是否为数字（包含汉字、阿拉伯数字等）更广泛
```

补充：需要十分注意的一点事，当我们将str转化为list时（如‘sfda’：转化为的是['s', 'f', 'd', 'a']，而不是[‘sfda’]）

### 2. 索引技巧

#### 2.1 列表:

`list.index()`

```python
# 返回第一个匹配元素的索引，如果找不到该元素则会引发 ValueError 异常

list.index(element, start, end)

my_list = [10, 20, 30, 40, 50, 30]
index = my_list.index(30)
print(index)  # 输出：2

index = my_list.index(30, 3)
print(index)  # 输出：5
```

```python
list(zip(a, b)) # a, b两列表，[1, 2, 4]; [1, 3, 4]=>[[1, 1], [2, 3], [4, 4]]
```

#### 2.2 字符串：

`str.find()` 和 `str.index()`

```python
my_string = "Hello, world!"
index = my_string.find("world")
print(index)  # 输出：7

index = my_string.find("Python")
print(index)  # 输出：-1

my_string = "Hello, world!"
index = my_string.index("world")
print(index)  # 输出：7

index = my_string.index("Python")  # 引发 ValueError
```

#### 2.3 字典:

```python
my_dict = {'a': 1, 'b': 2, 'c': 3}
exists = 'b' in my_dict
print(exists)  # 输出：True

exists = 'd' in my_dict
print(exists)  # 输出：False

keys_list = list(my_dict.keys())
index = keys_list.index('b')
print(index)  # 输出：1

# 直接查找字典中的键
index = list(my_dict).index('b')
print(index)  # 输出：1

dict.get(key, default=None)
# 返回指定键的值，如果值不在字典中返回default值
dict.setdefault(key, default=None)
# 和get()类似, 但如果键不存在于字典中，将会添加键并将值设为default
```

#### 2.4 集合

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}

# 并集
union_set = set1 | set2
print("并集:", union_set)  # 输出：{1, 2, 3, 4, 5}

# 交集
intersection_set = set1 & set2
print("交集:", intersection_set)  # 输出：{3}

# 差集
difference_set = set1 - set2
print("差集:", difference_set)  # 输出：{1, 2}

# 对称差集
symmetric_difference_set = set1 ^ set2
print("对称差集:", symmetric_difference_set)  # 输出：{1, 2, 4, 5}
```

## import相关

```python
# pylint: skip-file
import heapq
from collections import defaultdict
from collections import dequeue
import bisect
from functools import lru_cache
@lru_cache(maxsize=None)
import sys
sys.setrecursionlimit(1<<32)
import math
math.ceil()  # 函数进行向上取整
math.floor() # 函数进行向下取整。
math.isqrt() # 开方取整
exit()
```

```python
from collections import Counter
# 创建一个包含多个重复元素的列表/字典
data = [1, 2, 3, 4, 1, 2, 3, 1, 2, 1]
# 使用Counter函数统计各个元素出现的次数
counter_result = Counter(data)
print(counter_result)
#输出
Counter({1: 4, 2: 3, 3: 2, 4: 1})
```

#### bisect

1. **bisect.bisect_left(a, x, lo=0, hi=len(a))**
   - 在列表`a`中查找元素`x`的插入点，使得插入后仍保持排序。
   - 返回插入点的索引，插入点位于`a`中所有等于`x`的元素之前。
2. **bisect.bisect_right(a, x, lo=0, hi=len(a))** 或 **bisect.bisect(a, x, lo=0, hi=len(a))**
   - 类似于`bisect_left`，但插入点位于`a`中所有等于`x`的元素之后。
3. **bisect.insort_left(a, x, lo=0, hi=len(a))**
   - 在`a`中查找`x`的插入点并插入`x`，保持列表`a`的有序。
   - 插入点位于`a`中所有等于`x`的元素之前。
4. **bisect.insort_right(a, x, lo=0, hi=len(a))** 或 **bisect.insort(a, x, lo=0, hi=len(a))**
   - 类似于`insort_left`，但插入点位于`a`中所有等于`x`的元素之后。

###### 示例代码

```
python复制代码import bisect

a = [1, 2, 4, 4, 5]

# 查找插入点
print(bisect.bisect_left(a, 4))  # 输出: 2
print(bisect.bisect_right(a, 4)) # 输出: 4

# 插入元素
bisect.insort_left(a, 3)
print(a)  # 输出: [1, 2, 3, 4, 4, 5]

bisect.insort_right(a, 4)
print(a)  # 输出: [1, 2, 3, 4, 4, 4, 5]
```

#### 内置函数

```Python
sorted(iterable[, key[, reverse]]) # 返回值
list.sort([key[,reberse]])

print(*list)

lambda
aim_list = sorted(list, key = lambda o: o[1]) #举例
```

```python
python itertools.product(range(2), repeat=6) 生成6元元组，01的全排列
可用于: for l in itertools.product(range(n), repeat=(m))

```

## 转换

#### 进制

```python
b = bin(item)  # 2进制
o = oct(item)  # 8进制
h = hex(item)  # 16进制
```

#### ASCII

```python
ord(char) -> ASCII_value
chr(ascii_value) -> char
```

#### print保留小数

```python
print("%.6f" % x)
print("{:.6f}".format(result))
# 当输出内容很多时：
print('\n'.join(map(str, ans)))
```

## 算法

### 2.强联通子图

Kosaraju's算法可以分为以下几个步骤：

1. **第一次DFS**：对图进行一次DFS，并记录每个顶点的完成时间（即DFS从该顶点返回的时间）。
2. **转置图**：将图中所有边的方向反转，得到转置图。
3. **第二次DFS**：根据第一次DFS记录的完成时间的逆序，对转置图进行DFS。每次DFS遍历到的所有顶点构成一个强连通分量。

### 详细步骤

1. **第一次DFS**：
   - 初始化一个栈用于记录DFS完成时间顺序。
   - 对图中的每个顶点执行DFS，如果顶点尚未被访问过，则从该顶点开始DFS。
   - DFS过程中，当一个顶点的所有邻居都被访问过后，将该顶点压入栈中。
2. **转置图**：
   - 创建一个新的图，边的方向与原图相反。
3. **第二次DFS**：
   - 初始化一个新的访问标记数组。
   - 根据栈中的顺序（即第一步中记录的完成时间的逆序）对转置图进行DFS。
   - 每次从栈中弹出一个顶点，如果该顶点尚未被访问过，则从该顶点开始DFS，每次DFS遍历到的所有顶点构成一个强连通分量。

### 示例代码

以下是Kosaraju's算法的Python实现：

```python
from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.V = vertices

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def _dfs(self, v, visited, stack):
        visited[v] = True
        for neighbour in self.graph[v]:
            if not visited[neighbour]:
                self._dfs(neighbour, visited, stack)
        stack.append(v)

    def _transpose(self):
        g = Graph(self.V)
        for i in self.graph:
            for j in self.graph[i]:
                g.addEdge(j, i)
        return g

    def _fillOrder(self, v, visited, stack):
        visited[v] = True
        for neighbour in self.graph[v]:
            if not visited[neighbour]:
                self._fillOrder(neighbour, visited, stack)
        stack.append(v)

    def _dfsUtil(self, v, visited):
        visited[v] = True
        print(v, end=' ')
        for neighbour in self.graph[v]:
            if not visited[neighbour]:
                self._dfsUtil(neighbour, visited)

    def printSCCs(self):
        stack = []
        visited = [False] * self.V

        for i in range(self.V):
            if not visited[i]:
                self._fillOrder(i, visited, stack)

        gr = self._transpose()

        visited = [False] * self.V

        while stack:
            i = stack.pop()
            if not visited[i]:
                gr._dfsUtil(i, visited)
                print("")

# 示例使用
g = Graph(5)
g.addEdge(1, 0)
g.addEdge(0, 2)
g.addEdge(2, 1)
g.addEdge(0, 3)
g.addEdge(3, 4)

print("Strongly Connected Components:")
g.printSCCs()
```

### 二分查找

```python
# hi:不可行最小值， lo:可行最大值
lo, hi, ans = 0, max(lst), 0
while lo + 1 < hi:
    mid = (lo + hi) // 2
    # print(lo, hi, mid)
    if check(mid): # 返回True，是因为num>m，是确定不合适
        ans = mid
        lo = mid # 所以lo可以置为 mid + 1。
    else:
        hi = mid
#print(lo)
print(ans)
```

# 数据结构

### 单调栈
```python
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

### 并查集

```python
P = list(range(N))
def p(x):
    if P[x] == x:
        return x
    else:
        P[x] = p(P[x])
        return P[x]
    
def union(x, y):
    px, py = p(x), p(y)
    if px==py:
        return True
    else:
        if <不合法>:  # 根据题意，有时可略
            return False
        else:
            P[px] = py
            return True
```

### 8.Trie

1. **插入（Insert）**：

   ```python
   class TrieNode:
       def __init__(self):
           self.children = {}
           self.is_end_of_word = False
   
   class Trie:
       def __init__(self):
           self.root = TrieNode()
   
       def insert(self, word):
           node = self.root
           for char in word:
               if char not in node.children:
                   node.children[char] = TrieNode()
               node = node.children[char]
           node.is_end_of_word = True
   ```

2. **查找（Search）**：

   ```python
   def search(self, word):
       node = self.root
       for char in word:
           if char not in node.children:
               return False
           node = node.children[char]
       return node.is_end_of_word
   ```

3. **前缀查询（StartsWith）**：

   ```python
   def starts_with(self, prefix):
       node = self.root
       for char in prefix:
           if char not in node.children:
               return False
           node = node.children[char]
       return True
   ```

