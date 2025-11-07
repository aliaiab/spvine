#version 460

struct StrTest {
    int a;
    int b;
    int c;
};

#define u32 uint
#define i32 int
#define f32 float

#ifndef i32
#error Sus
#endif

#define ADD ad\
d

u32 ADD(f32 a, f32 b);

// //x##y => >tok_chain_start x, y tok_chain_end

// // typedef int32 int;

// \

int mul(int a, int b) {
    // int x;

    return add(a, b);
}

f32 add(f32 a, f32 b) {
    return a + 4;
}

void main() {
    // int a = 0;

    // int a = 0;

    // gl_Position = vec4(0);
}
