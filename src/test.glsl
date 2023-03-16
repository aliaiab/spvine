#version 450 
#define ADD +

#define TEST_FLAG 0

#if TEST_FLAG

#endif

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
}