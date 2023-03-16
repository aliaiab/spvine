#version 450 
#define ADD +
#define FLOAT FOLLL

#define TEST_FLAG 0

#if TEST_FLAG
#define TEST_FLAG_2 1
#define PI 3.1415926
#endif

#error "STINKIE"

//adds two numbers, a and b
float add(float a, float b) {
    return a ADD b;
}

/* 
    entrypoint procedure

    good luck!
*/
void main() {
    float res = add(1, 2 + 1);

    //clamp res to 1
    if (res > 1) {
        res = 1;
    }
    else {
        res = 0;
    }

    int res_as_int = (int)res;
}