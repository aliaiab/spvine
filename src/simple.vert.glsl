#version 460

struct StrTest {
    int a;
    int b;
    int c;
};

#define i32 int
#define f32 float

#ifndef i32
#error Sus
#endif

i32 add(i32 a, i32 b);
f32 add(f32 a, f32 b);
void main();

//x##y => >tok_chain_start x, y tok_chain_end

// typedef int32 int;

\

int add(i\nt a, int b) {
    int x;

    return a + b;
}

float add(float a, float b) {
    return a + b;
}

void main() {
    int a = 0;

    // int a = 0;

    // gl_Position = vec4(0);
}