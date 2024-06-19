#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <list>
#include <deque>
#include <set>
#include <algorithm>
using namespace std;


int main()
{
    system ("chcp 65001");
    vector<int> vec={2,43,5,7,3,3,65,6,7};
    sort(vec.begin(), vec.end());
    cout<<"sorted vector :";
    for(int i:vec){
        cout<<i<<" ";
    }
    cout<<endl;

    auto it=find(vec.begin(), vec.end(),3);//a到b有c嗎
    if(it==vec.end())cout<<"Not found";
    else cout<<"Have "<<*it<<endl;

    replace(vec.begin(), vec.end(), 3, 69);//把開始到結尾裡的a換成b

    vector<int> vec2(vec.size());//複製進去前要定義空間
    copy(vec.begin(), vec.end(), vec2.begin());
    cout<<"copy of vec: ";
    for(int i: vec2){
        cout<<i<<" ";
    }
    cout<<endl;

    int count69=count(vec2.begin(),vec2.end(),69);
    cout<<"vec2 have "<<count69<<" of 69"<<endl;
    return 0;
}