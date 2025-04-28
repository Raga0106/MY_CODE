/**
 * Briefly explain the function of this class.
 *
 * @author 			Your name here
 * @ID 				Your student ID here
 * @Department 		Engineering Science and Ocean Engineering
 * @Affiliation 	National Taiwan University
 *
 * Date.cpp
 * version 1.0
 */


#include "Date.h"

  /** 
   *  Constructs a Date with the given month, day and year.   If the date is
   *  not valid, the entire program will halt with an error message.
   *  @param month is a month, numbered in the range 1...12.
   *  @param day is between 1 and the number of days in the given month.
   *  @param year is the year in question, with no digits omitted.
   *
   *  Grade: 15%
   */
  Date::Date(int month, int day, int year) {
	        if (isValidDate(month, day, year)) {
            this->month = month;//利用指針設定this的月、日、年＝輸入值
            this->day = day;
            this->year = year;
        } else {
            cerr << "Error: Invalid date!" << endl;//立即顯示錯誤的原因
            exit(1);
        }
  }
  

  /** 
   *  Constructs a Date object corresponding to the given string.
   *  @param s should be a string of the form "month/day/year" where month must
   *  be one or two digits, day must be one or two digits, and year must be
   *  between 1 and 4 digits.  If s does not match these requirements or is not
   *  a valid date, the program halts with an error message.
   *
   *  Grade: 30%
   */
  Date::Date(const string& s) {
        stringstream ss(s);//把字串輸入至stringstream方便後面將字串轉換成數字
        string monthStr, dayStr, yearStr;

        getline(ss, monthStr, '/');//利用‘/’去分隔字串並輸入到對應的字串變數
        getline(ss, dayStr, '/');
        getline(ss, yearStr, '/');

        int month = stoi(monthStr);//將字串轉換成數字
        int day = stoi(dayStr);
        int year = stoi(yearStr);
        //  如果日期是合理的就用參數建構一個Date
        if (isValidDate(month, day, year)) {
            this->month = month;
            this->day = day;
            this->year = year;
        } else {
            cerr << "Error: Invalid date format or invalid date!" << endl;
            exit(1);
        }
  }


  /** 
   *  Checks whether the given year is a leap year.
   *  @return true if and only if the input year is a leap year.
   *
   *  Grade: 10%
   */
  bool Date::isLeapYear(int year) const{
        if (year % 4 != 0) return false;
        if (year % 100 != 0) return true;
        if (year % 400 == 0) return true;
        return false;                      // replace this line with your solution
  }


  /** 
   *  Returns the number of days in a given month.
   *  @param month is a month, numbered in the range 1...12.
   *  @param year is the year in question, with no digits omitted.
   *  @return the number of days in the given month.
   *
   *  Grade: 10%
   */
  int Date::daysInMonth(int month, int year) {
        if (month == 1 || month == 3 || month == 5 || month == 7 || month == 8 || month == 10 || month == 12)
            return 31;
        if (month == 2)
            return isLeapYear(year) ? 29 : 28;//如果是二月且閏年就回傳29反之28
        return 30;//除了上述月份回傳30                        // replace this line with your solution
  }


  /** 
   *  Checks whether the given date is valid.
   *  @return true if and only if month/day/year constitute a valid date.
   *
   *  Years prior to A.D. 1 are NOT valid.
   *
   *  Grade: 20%
   */
  bool Date::isValidDate(int month, int day, int year) {
        if (month < 1 || month > 12 || year < 1) return false;
        if (day < 1 || day > daysInMonth(month, year)) return false;
        return true;                   // replace this line with your solution
  }


  /** 
   *  Returns a string representation of this Date in the form month/day/year.
   *  The month, day, and year are expressed in full as integers; for example,
   *  10/17/2010 or 5/11/258.
   *  @return a String representation of this Date.
   *
   *  Grade: 20%
   */
  string Date::toString() {
        return to_string(month) + "/" + to_string(day) + "/" + to_string(year);                   // replace this line with your solution
  }


  /** 
   *  Determines whether this Date is before the Date d.
   *  @return true if and only if this Date is before d.
   *
   *  Grade: 10%
   */
  bool Date::isBefore(const Date& d) {
        if (year < d.year) return true;//當前年份小於給定則回傳true
        if (year == d.year && month < d.month) return true;//年份相同且月份小於回傳true
        if (year == d.year && month == d.month && day < d.day) return true;//年月都相同，日小於給定的回傳true
        return false;//上述都不成立則回傳false                       // replace this line with your solution
  }
  

  /** 
   *  Determines whether this Date is after the Date d.
   *  @return true if and only if this Date is after d.
   *
   *  Grade: 10%
   */
  bool Date::isAfter(const Date& d) {
        return !isBefore(d) && !isEqual(d);//非早於且非同日則回傳true                      // replace this line with your solution
  }
  
  
  /** 
   *  Determines whether this Date is equal to the Date d.
   *  @return true if and only if this Date is the same as d.
   *
   *  Grade: 10%
   */
  bool Date::isEqual(const Date& d) {
        return (year == d.year && month == d.month && day == d.day);
                      // replace this line with your solution
  }
  

  /** 
   *  Returns the number of this Date in the year.
   *  @return a number n in the range 1...366, inclusive, such that this Date
   *  is the nth day of its year.  (366 is only used for December 31 in a leap
   *  year.)
   *
   *  Grade: 15%
   */
  int Date::dayInYear()const {
        int monthsDay[12] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };//定義每個月有幾天，二月預設28天
        if (isLeapYear(year)) monthsDay[1] = 29;  // 如果該年是閏年，則把二月改成29天
        int sum = 0;
        for (int i = 0; i < month - 1; ++i) {
            sum += monthsDay[i];//計算該月前有幾天
        }
        return sum + day;//加上該月的第幾天                         // replace this line with your solution
  }
  

  /** Determines the difference in days between d and this Date.  For example,
   *  if this Date is 6/16/2006 and d is 6/15/2006, the difference is 1.
   *  If this Date occurs before d, the result is negative.
   *  @return the difference in days between d and this Date.
   *
   *  Grade: 10%
   */
  int Date::difference(const Date& d) {
        int days1 = dayInYear() + (year * 365) + (year / 4 - year / 100 + year / 400);//年份中的第幾天＋幾年份＊365+計算度過了幾個閏年補上
        int days2 = d.dayInYear() + (d.year * 365) + (d.year / 4 - d.year / 100 + d.year / 400);
        return days1 - days2;                         // replace this line with your solution
  }

