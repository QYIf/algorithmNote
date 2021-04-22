# c++ 转 java

## 输入篇

```java
//创建一个cin
Scanner cin = new Scanner (new BufferedInputStream(System.in));
//读一个整数：
int n = cin.nextInt(); //相当于 scanf(”%d”, &n); 或 cin >> n;
//读一个字符串：
String s = cin.next(); //相当于 scanf(”%s”, s); 或 cin >> s;
//读一个浮点数：
double t = cin.nextDouble(); //相当于 scanf(”%lf”, &t); 或 cin >> t;
```

## 输出篇

```java
//常见的输出方式
System.out.println(); //输出并换行
System.out.print(); //输出不换行
System.out.printf(); //像printf一样输出

//格式控制
DecimalFormat f = new DecimalFormat("#.00#");
//这里0指一位数字，#指除0以外的数字。
```

## STL函数篇

```java
Arrays.fill(); //类似于memset但只用用于一维
Arrays.sort(); //排序函数
//重新写排序方法
Comparator<Integer> cmp = new Comparator<Integer>() {
    @Override
    //默认升序
    public int compare(Integer o1, Integer o2) {
        return o1 - o2;
    }
};
Arrays.sort(arr, cmp);

//二分查找
Arrays.binarySearch(all,x);
```

## STL容器篇

```java
//vector
List<Integer> a = new LinkedList<>(); //vector<int> a;
a.add(i); // a.push_back
a.get(i); // a[i]
a.contains(i); // 查看a中是否包含i
a.set(i, j); //a[i] = j;
a.remove(i); //a.erase();
Collections.sort(); //排序
Collections.binarySearch(all,x); //二分查找
//多维
List<Integer>[] list = new List[100];
int a = src.nextInt(), b = src.nextInt();
if (list[a] == null) list[a] = new ArrayList<Integer>();

//stack
Stack<Integer> st = new Stack<Integer>(); //创建一个栈
st.add(x); // 将x入栈
st.pop(); //把栈顶元素出栈
st.isEmpty(); //判断栈是否为空
st.peek()); //获取栈顶元素

//queue
Queue<Integer> q = new LinkedList<>();
q.offer(x); // 将x入队列
q.poll(); // 将队尾元素出队
q.isEmpty(); // 判断队列是否为空
q.peek()); //获取队尾元素

//priority_queue
PriorityQueue<Integer> q = new PriorityQueue<Integer>((o1, o2) -> o1 - o2); //后半部分是用于比较
q.add(x); // 将x入堆
q.poll(); // 将堆顶元素出堆
q.isEmpty(); // 判断堆是否为空
q.peek()); //获取堆顶元素

//set
HashSet<Integer> hs = new HashSet<Integer>();
hs.add(v); //相当于insert
hs.contains(v); //相当于count != 0
    
//map
HashMap<Integer, Integer> mp = new HashMap<Integer, Integer>();
mp.put(i, j); // mp[i] = j;
mp.containsKey(i); // mp.count(i);
mp.get(i); // mp[i];
mp.replace(i, j); // mp[i] = j; (i已经存在)
```

## 大数篇

```java
//大数创建
BigInteger a = new BigInteger(src.next(), 10); //str表示数字, 10代表十进制
BigInteger b = new BigInteger(src.next(), 10); //str表示数字, 10代表十进制
//大数加法
System.out.println(a.add(b));
//大数减法
System.out.println(a.subtract(b));
//大数乘法
System.out.println(a.multiply(b));
//大数除法
System.out.println(a.divide(b));
//大数求余
System.out.println(a.remainder(b));
```

## 字符串篇

```java
//String
String a = src.next();
a.equals(b); //比较两个字符串是否相等
a.length(); //获取字符串长度
a.toCharArray(); //把字符串转化为一个字符数组
a.indexOf(str, idx); //在a中寻找从idx开始的子串str的下标
a = a + b; //合并两字符串
a.charAt(i); //获取字符串第i个字符
a.substring(i, j); //获取从i到j - 1 的子串
String.valueOf(); //数字转字符串

//StringBuilder
StringBuilder s = new StringBuilder("abcd");
s.append(i); //在s后面添加元素
s.delete(i, j); //删除i到j - 1的元素
s.insert(i, str); //在第i - 1个字符后面插入字符串str
```


## 基础算法篇

### 1. 离散化

```java
//二分查找对应元素下标
static int find(int x) {
    int l = 0 , r = all.size() - 1;
    while(l < r) {
        int mid = (l + r) >> 1;
        if(all.get(mid) >= x) r = mid;
        else l = mid + 1;
    }
    return l + 1;
}
//模拟C++的一个去重函数
static int unique() {
    int j = 0;
    for (int i = 0; i < all.size(); i++) {
        if (i == 0 || all.get(i) != all.get(i - 1)) all.set(j++, all.get(i));
    }
    return j;
}
//排序
Collections.sort(all);
//去重
int la = unique();
//把有用部分留下
all = all.subList(0, la);
```

## 数据结构篇

### 1.并查集

```java
//找父节点
static int find(int x) {
    if(p[x] == x) return x;
    int tmp = find(p[x]);
    d[x] += d[p[x]];
    return p[x] = tmp;
}

//合并两个集合
p[pa] = pb;
d[pa] = (d[b] - d[a] + c + 2) % 3;
```

### 2.trie树

```java
//插入
static void insert() {
    int p = 0;
    for(int i = 0; i < s.length(); i++) {
        if(tr[p][s.charAt(i) - 'a'] == 0) tr[p][s.charAt(i) - 'a'] = ++idx;
        p = tr[p][s.charAt(i) - 'a'];
    }
    cnt[p]++;
}
//查询
static int ask() {
    int p = 0;
    for(int i = 0; i < s.length(); i++) {
        if(tr[p][s.charAt(i) - 'a'] == 0) return 0;
        p = tr[p][s.charAt(i) - 'a'];
    }
    return cnt[p];
}
```

## Eclipse使用篇

### 1.快捷键

alt + shift + r :	重命名

ctrl + . 及c trl + 1:	下一个错误及快速修改

Ctrl + Shift + F:	格式化当前代码

Ctrl + Shift + O:	组织导入

### 2.一些问题的解决方式

如何打开控制台：

