#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <list>
#include <deque>
#include <set>
#include <iterator>
#include <algorithm>
using namespace std;


int main()
{
    //system ("chcp 65001");
    vector<int> vec1={3,4,65,73,2,5,6,7,8,9,10,};
    vector<int> vec2(vec1.size());
    sort(vec1.begin(), vec1.end());
    cout<<"vec1 have: ";
    //输入迭代器只能读取数据，支持单向移动。
    for(vector<int>::iterator it=vec1.begin(); it!=vec1.end();it++){//創建一個vector<int>的迭代器
        cout<<*it<<" ";
    }
    cout<<endl;
    //输出迭代器只能写入数据，支持单向移动。
    vector<int>::iterator it=vec2.begin();
    for(int i=1;i<=vec1.size();i++){
        *it=i*i;
        it++;
    }
    cout<<"vec2 have: ";
    for(const int &elem:vec2){//不可修改的參考
        cout<<elem<<" ";
    }
    return 0;
}