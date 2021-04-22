# C++ String 使用

## String 构造函数

```c++
a) string s; //生成一个空字符串 s
b) string s(str) //拷贝构造函数 生成 str 的复制品
c) string s(str,stridx) //将字符串 str 内“始于位置 stridx”的部分当作字符串的初值
d) string s(str,stridx,strlen) //将字符串 str 内“始于 stridx 且长度顶多 strlen”的部分作为字符串的初值
e) string s(cstr) //将 C 字符串作为 s 的初值
f) string s(chars,chars_len) //将 C 字符串前 chars_len 个字符作为字符串 s 的初值。
g) string s(num,c) //生成一个字符串，包含 num 个 c 字符
h) string s(beg,end) //以区间 beg;end(不包含 end)内的字符作为字符串 s 的初值
i) s.~string() //销毁所有字符，释放内存
```

## String 操作函数

```c++
    a) =,assign()   //赋以新值
    b) swap()   //交换两个字符串的内容
    c) +=,append(),push_back() //在尾部添加字符
    d) insert() //插入字符
    e) erase() //删除字符
    f) clear() //删除全部字符
    g) replace() //替换字符
    h) + //串联字符串
    i) ==,!=,<,<=,>,>=,compare()  //比较字符串
    j) size(),length()  //返回字符数量
    k) max_size() //返回字符的可能最大个数
    l) empty()  //判断字符串是否为空
    m) capacity() //返回重新分配之前的字符容量
    n) reserve() //保留一定量内存以容纳一定数量的字符
    o) [ ], at() //存取单一字符
    p) >>,getline() //从stream读取某值
    q) <<  //将谋值写入stream
    r) copy() //将某值赋值为一个C_string
    s) c_str() //将内容以C_string返回
    t) data() //将内容以字符数组形式返回
    u) substr() //返回某个子字符串
    v)查找函数
    w)begin() end() //提供类似STL的迭代器支持
    x) rbegin() rend() //逆向迭代器
    y) get_allocator() //返回配置器
```
