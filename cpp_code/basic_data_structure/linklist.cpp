#include <iostream>
//#include <cmath>
//#include <string>
//#include <vector>
//#include <map>
#include <list>
//#include <deque>
//#include <set>
//#include <iterator>
//#include <forward_list>
//#include <algorithm>
using namespace std;

struct Node{

    int data;
    Node* next;
};
class LinkList{
public:
    LinkList(): head(nullptr){}//这是构造函数的名称，当创建一个链表对象时会被调用。它用于在对象创建时初始化类的成员变量。在这里，它将链表的头节点（head）初始化为 nullptr。

    void insert(int value){//為lisklist創造一個從頭插入的函數

        Node* newNode=new Node{value,head};//new 關鍵字將分配足夠的記憶體空間來存儲一個新的 Node 物件。
        //newNode的值術數進去後next是原本的頭

        head=newNode;//所以頭是新的點
    }
    void print(){
        Node* current=head;
        while (current!=nullptr){
            cout<<current->data<<"->";
            current=current->next;
        }
        cout<<"null";
    }
    void removevalue(int value){
        Node* current=head;
        Node* previous=nullptr;
        while(current!=nullptr){
            if(current->data==value){
                if(previous==nullptr)head=current->next;
                else previous->next=current->next;
                current=current->next;
            }
            else{
            previous=current;
            current=current->next;
            }
        }
    }
private://private裡的東西只能在class的大括弧裡用
    Node*head;
};

int main()
{
    //system ("chcp 65001");
    class LinkList one;
    for(int i=0;i<10;i++){
        one.insert(i);
    }
    one.removevalue(6);
    one.print();

    return 0;
}