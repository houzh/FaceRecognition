#include"android_buf.hpp"
#include<android/log.h>
#include <strstream>
#include <iostream>
#define LOG "cout::AndroidBuf"
#define LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG,__VA_ARGS__)

AndroidBuf::AndroidBuf() {
    buffer_[BUFFER_SIZE] = '\0';
    setp(buffer_, buffer_ + BUFFER_SIZE - 1);
}
AndroidBuf::~AndroidBuf() {
    sync();
}

int AndroidBuf::flush_buffer() {
    int len = int(pptr() - pbase());
    if (len <= 0)
        return 0;

    if (len <= BUFFER_SIZE)
        buffer_[len] = '\0';

    LOGE("%s", buffer_);

    pbump(-len);
    return len;
}

std::streambuf::int_type AndroidBuf::overflow(std::streambuf::int_type c)
{
     if (c != EOF) {
         *pptr() = c;
         pbump(1);
     }
     flush_buffer();
     return c;
}

int AndroidBuf::sync() 
{
       flush_buffer();
       return 0;
}
