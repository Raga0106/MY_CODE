#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <list>
#include <deque>
#include <set>
using namespace std;

int main()
{
    system("chcp 65001");
    // vector在末尾插入和删除元素非常高效，但在中间插入和删除元素效率较低。
    vector<int> vec; // 動態調整大小的陣列
    int n;
    cout << "Enter num for vector:";
    while ((cin >> n))
    {
        vec.push_back(n); // 往後塞東西
        if (n == 0)
            break;
    }
    vec.pop_back(); // 拔最後一個
    cout << "一般的方法" << endl;
    for (int i = 0; i < vec.size(); i++)
    {
        cout << "At vec[" << i << "]=" << vec[i];
        if (i + 1 == vec.size())
            cout << endl;
        else
            cout << " , ";
    }
    cout << "迭代器" << endl;
    for (auto at = vec.begin(); at != vec.end(); at++)
    {
        cout << "Inside  vector have " << *at << endl;
    }

    // list是一个双向链表，支持在任何位置快速插入和删除元素，但不支持随机访问。
    list<int> lst;
    cout << "Enter num for list:";
    while ((cin >> n))
    {
        lst.push_back(n); // 往後塞東西
        if (n == 0)
            break;
    }
    cout << "I pop_back() the number of " << *prev(lst.end()) << endl;
    lst.pop_back();
    cout << "I pop_front() the number of " << *lst.begin() << endl;
    lst.pop_front();
    auto it = lst.begin();
    if (distance(it, lst.end()) > 3)
    {
        advance(it, 3); // 使指針位置前進數格
        cout << "I erase the number of " << *it << endl;
        lst.erase(it);
    }
    else
    {
        cout << "error" << endl;
    }

    lst.insert(it,2,10); // 在it的位置前插入2个10

    cout << "迭代器" << endl;
    for (auto at = lst.begin(); at != lst.end(); at++)
    {
        cout << "Inside  list have " << *at << endl;
    }
    // deque是一个双端队列，可以在两端快速插入和删除元素。
    // set是一个集合，存储唯一的元素，并自动排序。
    set<int> s;
    cout << "Enter num for set:";
    while ((cin >> n))
    {
        s.insert(n); //  重复元素不会插入
        if (n == 0)
            break;
    }
    if (s.find(69) != s.end()) // find會尋找東西的位置，如果沒有就會回傳最後一格(NULL)
    {
        cout << "69 is in set, know by find()" << endl;
    }
    else
        cout << "69 is not in set, know by find()" << endl;
    if (s.count(69)) // count會尋找東西，如果有就會回傳true
    {
        cout << "69 is in set, know by count()" << endl;
    }
    else
        cout << "69 is not in set, know by count()" << endl;
    // 遍历元素自动排序
    for (auto it = s.begin(); it != s.end(); ++it)
    {
        std::cout << "Element: " << *it << std::endl;
    }
    //map
    map<string,int> mp;
    //map[key] = value，如果該 key 值已經存在則 value 會被更新成新的數值
    return 0;
}