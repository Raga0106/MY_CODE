/**
 * 
 *
 * @author 			Your name here
 * @ID 				Your student ID here
 * @Department 		Engineering Science and Ocean Engineering
 * @Affiliation 	National Taiwan University
 *
 * TestDate.cpp
 * version 1.0
 */


#include "Date.h"


  int main() {
    cout << "\nTesting constructors->";
    Date* d1 = new Date(1, 1, 1);
    cout << "\nDate should be 1/1/1: " << d1->toString();
    delete d1;
	string s1 = "2/29/2000";
    d1 = new Date(s1);
    cout << "\nDate should be 2/29/2000: " << d1->toString();
    delete d1;
	s1 = "2/29/1904";
    d1 = new Date(s1);
    cout << "\nDate should be 2/29/1904: " << d1->toString();
    delete d1;
    d1 = new Date(12, 31, 1975);
    cout << "\nDate should be 12/31/1975: " << d1->toString();
	string s2 = "1/1/1976";
    Date* d2 = new Date(s2);
    cout << "\nDate should be 1/1/1976: " << d2->toString();
	string s3 = "1/2/1976";
    Date* d3 = new Date(s3);
    cout << "\nDate should be 1/2/1976: " << d3->toString();

    string s4 = "2/27/1977";
	Date* d4 = new Date(s4);
	string s5 = "8/31/2110";
    Date* d5 = new Date(s5);

    /* I recommend you write code to test the isLeapYear function! */

    cout << "\nTesting before, after and equal->";
    cout << "\n" << d2->toString() << " after " << d1->toString() << " should be true: " << d2->isAfter(*d1);
    cout << "\n" << d3->toString() << " after " << d2->toString() << " should be true: " << d3->isAfter(*d2);
    cout << "\n" << d1->toString() << " after " << d1->toString() << " should be false: " << d1->isAfter(*d1);
    cout << "\n" << d1->toString() << " after " << d2->toString() << " should be false: " << d1->isAfter(*d2);
    cout << "\n" << d2->toString() << " after " << d3->toString() << " should be false: " << d2->isAfter(*d3);

    cout << "\n" << d1->toString() << " before " << d2->toString() << " should be true: " << d1->isBefore(*d2);
    cout << "\n" << d2->toString() << " before " << d3->toString() << " should be true: " << d2->isBefore(*d3);
    cout << "\n" << d1->toString() << " before " << d1->toString() << " should be false: " << d1->isBefore(*d1);
    cout << "\n" << d2->toString() << " before " << d1->toString() << " should be false: " << d2->isBefore(*d1);
    cout << "\n" << d3->toString() << " before " << d2->toString() << " should be false: " << d3->isBefore(*d2);
                       
    cout << "\n" << d2->toString() << " equal " << d3->toString() << " should be false: " << d2->isEqual(*d3);
    cout << "\n" << d5->toString() << " equal " << d5->toString() << " should be true: " << d5->isEqual(*d5);
    cout << "\n" << d1->toString() << " equal " << d4->toString() << " should be false: " << d1->isEqual(*d4);

    cout << "\nTesting difference->";
    cout << "\n" << d1->toString() << " - " << d1->toString() << " should be 0: " << d1->difference(*d1);
    cout << "\n" << d2->toString() << " - " << d1->toString() << " should be 1: " << d2->difference(*d1);
    cout << "\n" << d3->toString() << " - " << d1->toString() << " should be 2: " << d3->difference(*d1);
    cout << "\n" << d3->toString() << " - " << d4->toString() << " should be -422: " << d3->difference(*d4);
    cout << "\n" << d5->toString() << " - " << d4->toString() << " should be 48762: " << d5->difference(*d4);
    cout << "\n";
  }