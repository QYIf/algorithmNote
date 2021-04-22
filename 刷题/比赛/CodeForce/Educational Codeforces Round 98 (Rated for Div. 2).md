# Educational Codeforces Round 98 (Rated for Div. 2)                

> 比赛心得：第一次打教育场，果然教育场就是被教育的。这场比赛感觉前面简单题的时间还是稍微用得太多了点，而且比较神奇的是此次b题的难度竟然比c题打qaq，其实大方向上我们都有带你想法，但是没有把众多性质合并在一起，因此次题我们没有写出来。下次在讨论时我们可以将一些本题的性质先列出来，然后经行统一整合说不定这道题我们可以做得出来。

## A. Robot Program

> 题目大意：这道题的意思是从（0，0）点走到（x，y）点，每次都可以进行上下左右和原地不动5种操作，每种操作不能连续执行两次，问最少要执行多少步才能到达我们要的地方

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
#define ll long long
using namespace std;
const int N = 200010;

int t, a, b; 
 
int main() {
	scanf("%d", &t);
	while(t--) {
		scanf("%d%d", &a, &b);
		if(a == b) printf("%d\n", a + b);
		else printf("%d\n", max(a, b) * 2 - 1);
	}
}
```

> 大致思路：这道题可以把两个坐标分成两个方向来看，每次都执行一次水平操作，一次垂直操作。如果坐标的x与y相等，那么我们只要把x和y坐标加起来即可，但是不想等情况可能会有停顿，如果我们已经到达了终点就不需要停顿了，因此在其他情况下我们取x和y中的最多值，使其乘2减1就是答案。

## B. Toy Blocks

> 大致题意：这道题的意思是有几个箱子，每个箱子都有一些木块。现在我们要在选择任意一个箱子时，都会拿出这个箱子的全部木块，然后把这些木块分配给其他剩余的几个箱子，使得这些箱子里面的木块个数相等（任意选一个箱子进行该操作都要满足此条件），但是有时候并不能做到这一点，此时我们需要给某些箱子来添加木块以做到这点，问至少要加多少个木块。

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
#define ll long long
using namespace std;
const int N = 200010;

ll t, n;

int main() {
	scanf("%lld", &t);
	while(t--) {
		ll sum = 0, mx = 0, ans = 0;
		scanf("%lld", &n);
		for(ll i = 0; i < n; i++) {
			ll tmp;
			scanf("%lld", &tmp);
			mx = max(tmp, mx);
			sum += tmp;
		}
		ans = mx * (n - 1) - sum;
        //小于0的情况下要补
		if(ans < 0) ans += ((-ans - 1) / (n - 1) + 1) * (n - 1);
		if(n == 2) puts("0");
		else printf("%lld\n", ans);
	}
}
```

> 大致思路：
>
> 将其中一个盒子的玩具数转移到剩下n-1个盒子后，这n-1个盒子玩具数相同，说明**玩具的总数必须是n-1的倍数**
> 因为任选一个盒子，都要有方案，假设n个盒子中第**盒子玩具数最多的盒子个数为x**，如果我们选择了其它盒子进行分配，则n - 1个盒子玩具数起码为x * (n - 1)
> 由此答案已经得出：
> 玩具的总数必须是n-1的倍数，且至少大于x * (n - 1)，然后计算当前玩具数，如果当前玩具数少于x * (n - 1），补上即可，若果多于，则将玩具数补至（x + 1）*（n - 1）个即可

## C. Two Brackets

> 大致题意：括号匹配，出现左括号以后出现一个右括号即能匹配出一对括号

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
#define ll long long
using namespace std;
const int N = 200010;

int t;
string bra;

int main() {
	scanf("%d", &t);
	while(t--) {
		cin >> bra;
		int l1 = 0, l2 = 0, ans = 0;
		for(int i = 0; i < bra.size(); i++) {
			if(bra[i] == '[') l1++;
			else if(bra[i] == '(') l2++;
			else if(bra[i] == ')') {
				if(l2 > 0) {
					ans++;
					l2--;
				}
			} else {
				if(l1 > 0) {
					ans++;
					l1--;
				}
			}
		}
		printf("%d\n", ans);
	}
}
```

> 大致思路：
>
> 用两个计数器来统计左括号出现的次数，当遇到右括号时，看看左边是否存在能与右括号匹配的左括号，有则ans++，同时把让相应括号的计数器减一。

## D. Radio Towers

> 大致题意：这道题首先有编号从0到n + 1的n + 2个塔，每个塔都有一个辐射范围现在要求符合以下规律的方案的概率。
>
> 要求如下：
>
> 0 和 n + 1 两个塔不能被覆盖
>
> 信号与信号之间不能重叠
>
> 其他的塔都要被信号覆盖到

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
#define ll long long
using namespace std;
const int N = 200010;
const ll M = 998244353;

ll n, f[N];

ll qmul(ll a, ll p) {
	ll m = 1;
	while(p) {
		if(p & 1) m = (m * a) % M;
		a = (a * a) % M;
		p /= 2;
	}
	return m % M;
}

int main() {
	scanf("%lld", &n);
	f[1] = f[2] = 1;
	for(ll i = 3; i <= n; i++) f[i] = (f[i - 1] + f[i - 2]) % M;
	ll ans = f[n] * qmul(qmul(2, n), M - 2) % M;
	printf("%lld\n", ans);
}
```

> 大致思路：这道题就是求符合要求的方案数，而我们枚举以下前面的方案就知道，方案数符合斐波那契数列，因此答案其实就是对应n的斐波那契数除以2^n的逆元即为答案。