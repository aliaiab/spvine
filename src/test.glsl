#version 450 
#define ADD +
#define FLOAT FOLLL

#define TEST_FLAG 0

#if TEST_FLAG
#define TEST_FLAG_2 1
#define PI 3.1415926
#endif

#define ENTRYPOINT main()

#define ENTRY ENTRYPOINT
#define ENTRY_ALIAS ENTRY

//adds two numbers, a and b
float add(float a, float b) {
    return a ADD b;
}

/* 
    entrypoint procedure

    good luck!
*/
void ENTRY_ALIAS {
    float res = add(1, 2 + 1);

    //clamp res to 1
    if (res > 1) {
        res -= 1;
    }
    else {
        res += 2;
    }

    int res_as_int = int(res);

    res_as_int *= 3;

    res_as_int = res_as_int * 10;

    res_as_int /= 2 + 3;

    /*
    type material: struct 
        %0 = field("a");
        %1 = field("b");

    proc add:
        %0 = arg(0);
        %1 = arg(1);

        %2 = add(%0, %1);
        %3 = ret(%2);

    proc main:
        %0 = proc_call(add, 1, 3); //%0 = add(1, 3);
        %1 = constant(1); //%1 = 1;
        %2 = cmp_gt(%0, %1); //%0 > %1
        %3 = if(%2);
        %4 = else;
        %5 = ret;
    */
}