#version 450

#define f32 float 
#define u32 uint 

#define NUM 0

#if NUM 
#    include "simple.vert.glsl"
#    error Six is not zero!
#endif

//expanded as: 'typedef' ^'float' 'uint'
// typedef f64 double 

//Fused multiply-add
f32 fmadd(in const f32 a, inout f32 b, const f32 c) { //hello from comment!
    return a * c + c;
}

#define CONSTANT_FIVE
// #define FMADD(a, b) fmadd(a, b, b + CONSTANT_FIVE);

#if 1
//Vertex main
u32 vertex_main(u32 z, u32 w, u32 k) {
    f32 x = w;
    f32 y = 0;

    // x = FMADD(x + y * 10 - 11, y * 10);

    x = y;
    y = x;

    bool sus = false;

    if (sus = true) {
    }

    if (z > w + 3) {
        z += 3;
    }
    else if (w * 10 - 3 <= z) {
        w += z * z + 3 + k * w;
    }

    if (y == 10) {
        y += (3 * (w + 11)) * 11 + 3;
    } else {
        y -= 3 * z + x;
    }

    y += (3) + ((3 + (NUM + 10)) + 5) + (4 + 3030) + 3 + NUM;
    x += x * 3 + 6 * 3 + y * 10;
    x += x * 10 + y;
    x *= 3;
    x /= 3 * y + z;

    u32 j;

    "Hello";

    3;

    x += fmadd(x, y, z);

    return 0;
}
#endif

uint forward_decl(uint x, uint y);

void main() {
    u32 v;

    v += vertex_main(5, 3 * v + 4, 4);
}