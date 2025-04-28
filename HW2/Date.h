/**
 * Briefly explain the function of this class.
 *
 * @author        
 * @ID            
 * @Department         Engineering Science and Ocean Engineering
 * @Affiliation     National Taiwan University
 *
 * Date.h
 * version 1.0
 */

#ifndef DATE_H
#define DATE_H

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace std;

class Date {
   private:
    int month, day, year;

   public:
    Date(int month, int day, int year);
    Date(const string& s);
    bool isLeapYear(int year)const;
    int daysInMonth(int month, int year);
    bool isValidDate(int month, int day, int year);
    string toString();
    bool isBefore(const Date& d);
    bool isAfter(const Date& d);
    bool isEqual(const Date& d);
    int dayInYear() const; 
    int difference(const Date& d);
};

#endif
