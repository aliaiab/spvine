#version 450
#define ADD +
#define FLOAT FOLLL

#define TEST_FLAG 0

#if TEST_FLAG
#define TEST_FLAG_2 1
#define PI 3.1415926
#endif

#ifndef TEST_FLAG_2 
#error TEST_FLAG_2 is sus.
#endif

#define ENTRYPOINT main()

#define ENTRY ENTRYPOINT
#define ENTRY_ALIAS ENTRY

//adds two numbers, a and b
float add(float a, float b) { //body
    float c = a + b;
    c += 1;
    c *= 2;
    c -= 1;
    return c;
}

/* 
    entrypoint procedure
*/
void ENTRY {
    float res = add(1, 2 + 1); //sus

    //clamp res to 1
    if (res > 1) {
        res -= 1;
    }
    else {
        res += 2;
    }

    int res_as_int = int(res); /* here is an cast to int */

    res_as_int *= 3;
    res_as_int = res_as_int * 10;
    res_as_int /= 2 + 3;

    gl_Position = vec4(0);
}