#ifndef __ANDROID_BUF_H__
#define __ANDROID_BUF_H__
#include <iostream>
#include <streambuf>
class AndroidBuf : public std::streambuf {
    enum {
        BUFFER_SIZE = 512,
    };

public:
    AndroidBuf();

    ~AndroidBuf();

protected:
    virtual std::streambuf::int_type overflow(std::streambuf::int_type c);
    virtual int sync(); 

private:
    int flush_buffer();

private:
    char buffer_[BUFFER_SIZE + 1];
};
#endif
