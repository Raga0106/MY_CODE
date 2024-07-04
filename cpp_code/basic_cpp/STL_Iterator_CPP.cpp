#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <list>
#include <deque>
#include <set>
#include <iterator>
#include <forward_list> //僅適用單項迭代
#include <algorithm>
using namespace std;

int main()
{
    system("chcp 65001");
    vector<int> vec1 = {
        3,
        4,
        65,
        73,
        2,
        5,
        6,
        7,
        8,
        9,
        10,
    };
    vector<int> vec2(vec1.size());
    sort(vec1.begin(), vec1.end());
    cout << "vec1 have: ";
    // 输入迭代器只能读取数据，支持单向移动。
    for (vector<int>::iterator it = vec1.begin(); it != vec1.end(); it++)
    { // 創建一個vector<int>的迭代器
        cout << *it << " ";
    }
    cout << endl;
    // 输出迭代器只能写入数据，支持单向移动。
    vector<int>::iterator it = vec2.begin();
    for (int i = 1; i <= vec1.size(); i++)
    {
        *it = i * i;
        it++;
    }
    cout << "vec2 have: ";
    for (const int &elem : vec2)
    { // 不可修改的參考
        cout << elem << " ";
    }
    cout << endl;

    // 前向迭代器可以读写数据，支持单向移动，允许多次遍历。
    forward_list<int> fowlist = {2, 0, 0, 5, 0, 1, 0, 6};
    for (forward_list<int>::iterator it = fowlist.begin(); it != fowlist.end(); it++)
    {
        *it += 1;
        *it *= 2;
    }
    cout << "fowlist have: ";
    for (const int &elem : fowlist)
    { // 不可修改的參考
        cout << elem << " ";
    }
    cout << endl;

    list<int> lst = {9, 4, 5, 3, 0, 9, 8, 7}; // list可以用双向迭代器可以读写数据，支持双向移动。
    list<int> lst2 = {6969, 8787};
    cout << "forward of lst: ";
    for (list<int>::iterator it = lst.begin(); it != lst.end(); it++)
    {
        cout << *it << " ";
    }
    cout << endl
         << "backward of lst: ";
    for (list<int>::reverse_iterator it = lst.rbegin(); it != lst.rend(); it++)
    { // 反向迭代器+1會往頭走
        cout << *it << " ";
    }
    cout << endl;
    // 隨機訪問迭代器就是直接訪問任和vector的位置例如vec[5]

    // 迭代器适配器
    copy(lst.begin(), lst.end(), front_inserter(lst2)); // lst插入lst2的前方
    cout << "lst插入lst2的前方" << endl;
    for (const int &elem : lst2)
    {
        cout << elem << " ";
    }
    cout << endl;
    cout << "lst插入lst2的後方" << endl;
    copy(lst.begin(), lst.end(), back_inserter(lst2)); // lst插入lst2的後方
    for (const int &elem : lst2)
    {
        cout << elem << " ";
    }
    cout << endl;
    auto it2 = lst2.begin();
    advance(it2, 1);                                   // 将迭代器移动到第二个位置
    copy(lst.begin(), lst.end(), inserter(lst2, it2)); // lst插入lst2的 第二鬲
    cout << "lst插入lst2的 第二鬲" << endl;
    for (const int &elem : lst2)
    {
        cout << elem << " ";
    }
    return 0;
}