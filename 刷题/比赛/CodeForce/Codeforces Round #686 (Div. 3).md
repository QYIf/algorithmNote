# [Codeforces Round #686 (Div. 3)](http://codeforces.com/contest/1454)

> 打div3的最大感受就是有时感觉一道题非常简单，但是整理思路又是会用半天。感觉这与自己对数据结构的不熟悉有关。感觉自己还是要对一些常见的操作进行一些归纳。

## [A. Special Permutation](http://codeforces.com/contest/1454/problem/A)

> 大致题意：所有的输一个第 i 位不等于 i 的一个数组

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
#define ll long long
using namespace std;

int n, t;

int main() {
	scanf("%d", &t);
	while(t--) {
		scanf("%d", &n);
		printf("%d ", n);
		for(int i = 1 ; i <= n - 1; i++) {
			printf("%d ", i);
		}
		printf("\n");
	}
	return 0;
}
```

> 先输出最大的值，然后从一到最后输出即可形成错位

## [B. Unique Bid Auction](http://codeforces.com/contest/1454/problem/B)

> 大致题意：这道题是要我们从一个数组中找到最小的且不重复的数字的下标。

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
#include<set>
#define ll long long
using namespace std;
const int N = 200010;

int n, t, cnt[N], arr[N];

int main() {
	scanf("%d", &t);
	while(t--) {
		int mn = -1, ans;
		memset(cnt, 0, sizeof cnt);
		memset(arr, 0, sizeof arr);
		scanf("%d", &n);
        //输入同时计数
		for(int i = 1; i <= n; i++) {
			scanf("%d", &arr[i]);
			cnt[arr[i]]++;
		}
        //找出现一次的最小数
		for(int i = 1; i <= n; i++) {
			if(cnt[i] == 1) {
				mn = i;
				break;
			}
		}
        //找下标
		for(int i = 1; i <= n && mn != -1; i++) {
			if(arr[i] == mn) {
				ans = i;
				break;
			}
		}
		if(mn == -1) puts("-1");
		else printf("%d\n", ans);
	}
	return 0;
}
```

>首先输入时统计数字出现的个数，然后从小到大找到第一个出现次数为1的数字，然后找到这个数字在数组中的下标

## [C. Sequence Transformation](http://codeforces.com/contest/1454/problem/C)

> 大致题意：这道题我们要先选择一个保留数，然后区间删除，但是选择的保留数不能被删掉。问选哪个保留数可以使得操作数尽可能的少。

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
#include<set>
#define ll long long
using namespace std;
const int N = 200010;

int n, t, cnt[N], arr[N];

int main() {
	scanf("%d", &t);
	while(t--) {
		scanf("%d", &n);
        //初始化
		int f = 1, mn = 0x3f3f3f3f;
		memset(cnt, 0, sizeof cnt);
		memset(arr, 0, sizeof arr);
		for(int i = 1; i <= n; i++) {
			scanf("%d", &arr[i]);
		}
        //特判只有一个数字的情况
		if(n == 1)  {
			puts("0");
			continue;
		}
		for(int i = 1; i <= n; i++) {
            //一开始时由于前面没有数字我们只要看后面即可
			if(i == 1) {
				if(arr[i] != arr[i + 1]) {
					cnt[arr[i]]++;
				}
                //最后一个数时，如果前面出现过这个数，说明这个数字前面的区间已经处理过了
                //如果这个数等于前一个且该数没处理过，说明前n - 1个数都是相等的因此不处理
			} else if(i == n) {
				if(cnt[arr[i]] == 0 && arr[i] != arr[i - 1]) {
					mn = 1;
					f = 0;
					break;
				}
                //中间部分如果当前数字没有被处理过，说明这个数需要处理前面和后面部分，如果处理过，则处理后面部分即可
			} else {
				if(cnt[arr[i]] == 0) {
					if(arr[i] != arr[i + 1]) cnt[arr[i]]++;
					if(arr[i] != arr[i - 1]) cnt[arr[i]]++;
				} else {
					if(arr[i] != arr[i + 1]) cnt[arr[i]]++;
				}
			}
		}
        //判断当前还有没有必要寻找最值
		if(f) {
			for(int i = 1; i <= n; i++) {
				if(mn > cnt[i] && cnt[i] != 0) {
					mn = cnt[i];
				}
			}
            //说明整个序列都相等，不用处理
			if(mn == 0x3f3f3f3f) mn = 0;
		}
		printf("%d\n", mn);
	}
	return 0;
}
```

> 这道题其实就是这道我们要分析每种数字的保留的情况下要删除的次数。具体统计方式见代码

## [D. Number into Sequence](http://codeforces.com/contest/1454/problem/D)

> 大致题意：这道题其实就是给定一个n找到一个总乘积为n的一个序列。该序列满足前一个数是后一个数的因子，且这个序列要尽可能的长。

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#define ll long long
using namespace std;
ll x, t, n;

int main() {
	scanf("%lld", &t);
	while(t--) {
		scanf("%lld", &n);
		ll ans, pr, mxc = 0, i = 2;
        //筛选因子
		while(i * i <= n) {
			ll tmp = n, c = 0;
            //求出因子个数并维护最大值
			while(tmp % i == 0) {
				c++;
				tmp /= i;
			}
			if(c > mxc) {
				mxc = c;
				pr = i;
				ans = tmp * i;
			}
			i++;
		}
        //判断当前数是否为素数，根据情况不同
		if(mxc > 1) {
			printf("%lld\n", mxc);
			for(ll i = 1; i < mxc; i++) {
				printf("%lld ", pr);
			}
			printf("%lld\n", ans);
		} else {
			printf("1\n%lld\n", n);
		}
	}
}
```

> 找出这个数的因子中包含数最多的一个，设这个数包含n个该因子，那么输出n - 1个该因子，然后输出被除以该因子以后剩下的数字即可。如果是素数则输出本身即可。