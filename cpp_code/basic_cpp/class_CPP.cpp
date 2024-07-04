#include <iostream>
#include <string>
using namespace std;

class car
{
public: // 定义类的公共成员，公共成员可以在类外部访问
    // 屬性
    string brand;
    string color;
    float speed;

    // 構造函數，這個函數可以創造這個class
    car(string b, string c, float s) : brand(b), color(c), speed(s) {}

    // 析構函數，在对象生命周期结束时自动调用，无需手动调用，每当一个Car对象被销毁时，析构函数将自动运行并输出信息。
    ~car()
    {
        cout << "deleted" << endl;
    }

    // 方法
    void what_is_this()
    {
        cout << "this is a " << color << " " << brand << " ruuning at " << speed << "km/h." << endl;
    }
};

class animal
{
public:
    void eat()
    {
        cout << "This animal is eating" << endl;
    }
    virtual void makeSound()
    { // 虚函数允许子类重写父类的方法，当我们通过父类指针或引用调用该方法时，实际调用的是子类的实现。
        cout << "Some generic animal sound." << endl;
    }
};

class dog : public animal // Dog类继承自Animal类，Dog类将拥有Animal类的所有公共成员。
{
public:
    void bark()
    {
        cout << "The dog is barking" << endl;
    }
    void makeSound() override // 覆蓋父類的同一個函數
    {
        cout << "WOLf" << endl;
    }
};

class cat : public animal // Dog类继承自Animal类，Dog类将拥有Animal类的所有公共成员。
{
public:
    void meow()
    {
        cout << "The cat is meowing" << endl;
    }
    void makeSound() override // 覆蓋父類的同一個函數
    {
        cout << "Meow" << endl;
    }
};

void playSound(animal &animal) // 通过父类指针或引用调用该方法时，实际调用的是子类的实现。
{
    animal.makeSound();
}

int main()
{
    car car1("BWM", "RED", 129);

    car1.what_is_this();
    car1.what_is_this();

    dog dog1;
    dog1.eat();
    dog1.bark();

    cat cat1;
    animal animal1;
    playSound(animal1);
    playSound(dog1);
    playSound(cat1);
    return 0;
}