#include <iostream>
#include <string>
#include <vector>
#include <map>
using namespace std;


int main()
{   
    vector<int> vec;//動態調整大小的陣列
    int n;
    cout<<"Enter num:";
    while((cin>>n)&&n!=0){
        vec.push_back(n);//往後塞東西
    }
    for(int i=0;i<vec.size();i++){
        cout<<"At vec["<<i<<"]="<<vec[i]<<endl;
    }





    return 0;
}