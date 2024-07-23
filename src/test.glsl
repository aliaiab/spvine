#version 450
#define ADD +
#define FLOAT FOLLL

// #include <std>
// #include "hello, world"
#define TEST_FLAG 0

#if TEST_FLAG
#define TEST_FLAG_2 1
#define PI 3.1415926
#endif

#ifndef TEST_FLAG_2 
#error TEST_FLAG_2 is sus.
#endif

// #undef TEST_FLAG_2

#define ENTRYPOINT main()

#define ENTRY ENTRYPOINT
#define ENTRY_ALIAS ENTRY

#define FUNC(x, y) x + y

//#line 10 "sus.h"

// #define i32 int
#define u32 uint
#define f32 float

// const u32 x = 2;

// struct IInt64 {
//     i32 val;
//     // ;
// };

// struct IUInt64 {
//     // IInt64 vali;
//     // u32 valu;
// };

//adds two numbers, a and b
float add(float a, float x) { //body
    //variable decl
    float f;

    //variable asign
    f = 3;

    //variable init
    float c = a + b; float n = 3;

    if (c) {
        c = c + 3;

        {
            int scoped = 3;
        
            c = c + 10;
            f = c + 10;
        }
    } else {
        c = 10;
        c = 2 + c;
        f = c + 1;
    }

    if (true) c = 3;

    // c = 3;
    // c += 3;
    // c -= 3;
    // c *= 3;

    float e = c + b;
    float d = e + b;

    return c + 3;
}

void meme() {}

void sus(float a) {
    return a;
}

float sub(float x, float y, u32 z) {
    float d = z;

    return x + y;
}

//-DCOMPILE_ENTRY 1

#define COMPILE_ENTRY 0
#define OOPS 0

#if OOPS
#error "Oops"
#endif 

/* 
    entrypoint procedure
*/
#if COMPILE_ENTRY
void ENTRY {
    float res = add(1, 2 + 1); //sus

    //clamp res to 1
    if (res == 1) {
        res -= 1;
    }
    else {
        res += 2;
    }

    #define BOOL_TRUE true

    u32 res_as_int = u32(BOOL_TRUE); /* here is an cast to int */

    res_as_int *= 3;
    res_as_int = res_as_int * 10;
    res_as_int /= 2 + 3;

    gl_Position = vec4(0);
}
#endif