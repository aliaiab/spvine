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

#define i32 int
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
f32 add(f32 a, f32 x) { //body
    //variable decl
    f32 f;

    //variable asign
    f = 3;

    //variable init
    f32 c = a + b; f32 n = 3;

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

    f32 e = c + b;
    f32 d = e + b;

    return c + 3;
}

void meme() {}

void sus(f32 a) {
    return a;
}

f32 sub(f32 x, f32 y, u32 z) {
    f32 d = z;

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
    f32 res = add(1, 2 + 1); //sus

    //clamp res to 1
    if (res > 1) {
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