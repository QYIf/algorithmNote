## A

## B.	Beauty Values

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#define ll long long
using namespace std;
const int N = 200010;

ll num, qty[N], ans, n;

int main() {
	scanf("%lld", &n);
	for(ll i = 1; i <= n; i++) {
		scanf("%lld", &num);
		ans += (n - i + 1) * (i - qty[num]);
		qty[num] = i;
	}
	printf("%lld", ans);
	return 0;
}
//
```

## C.	CDMA

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
#define ll long long
using namespace std;
const int N = 2500;

int arr[N][N], n;

int main() {
	scanf("%d", &n);
	arr[1][1] = 1;
	arr[1][2] = 1;
	arr[2][1] = 1;
	arr[2][2] = -1;
	for(int len = 2; len < n; len *= 2) {
		for(int i = 1; i <= len; i++) {
			for(int j = 1; j <= len; j++) {
				arr[i][j + len] = arr[i][j];
				arr[i + len][j] = arr[i][j];
				arr[i + len][j + len] = -arr[i][j];
			}
		}
	}
	for(int i = 1; i <= n; i++) {
		for(int j = 1; j <= n; j++) {
			if(j < n) printf("%d ", arr[i][j]);
			else printf("%d\n", arr[i][j]);
		}
	}
}
//这道题，只要每次将自身复制，放到左方和下方，然后把自己的相反数放到右下角，一直重复这样的操作即可。
```

## D

## E

## F.	Fraction Comparision

```cpp
#include<cstdio>
#include<cstring>
#include<algorithm>
#define ll long long
using namespace std;
ll x, y, a, b;

int main() {
	while(scanf("%lld%lld%lld%lld", &x, &a, &y, &b) != EOF) {
		if(x / a > y / b) printf(">\n");
		else if(x / a < y / b) printf("<\n");
		else {
			if((x % a) * b > (y % b) * a) {
				printf(">\n");
			} else if((x % a) * b < (y % b) * a) {
				printf("<\n");
			} else {
				printf("=\n");
			}
		}
	}
}
//本题需要注意考点在于这两个整数除法，直接除是没有余数的，但比较时需要判断余数，所以我们需要对余数的地方进行特殊的判断。
```

## G.	Gemstones

```cpp
#include<iostream>
#include<cstdio>
#include<cstring>
#include<algorithm>
#include<cmath>
#define ll long long
using namespace std;
const int N = 100010;

char st[N];
int top = 0, ans = 0;
string str;

int main() {
	cin >> str;
	for(int i = 0; i < str.size(); i++) {
		if(top >= 3) {
			if(st[top - 1] == st[top - 2] && st[top - 2] == st[top - 3]) {
				top -= 3;
				st[top++] = str[i];
				ans++;
			} else {
				st[top++] = str[i];
			}
		} else {
			st[top++] = str[i];
		}
	}
	if(st[top - 1] == st[top - 2] && st[top - 2] == st[top - 3]) ans++;
	printf("%d", ans);
}
```

## H

## I

## J.	

## K.	Random Point in Triangle

```cpp

```

