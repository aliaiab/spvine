#version 450

#define f32 float
#define u32 uint

#define NUM 0

// #if NUM
// #    include "simple.vert.glsl"
// #    error Six is not zero!
// #endif

//expanded as: 'typedef' ^'float' 'uint'
// typedef f64 double

//Fused multiply-add
f32 fmadd(const f32 a, const f32 b, const f32 c) { //hello from comment!
    return a * c + c;
}

#define CONSTANT_FIVE
// #define FMADD(a, b) fmadd(a, b, b + CONSTANT_FIVE);

f32 vertex_main(u32 z, u32 w, u32 k);

#if 0
//Vertex main
u32 vertex_main(u32 z, u32 w, u32 k) {
    f32 x = w;
    f32 y = 0;

    // x = FMADD(x + y * 10 - 11, y * 10);

    x = y;
    y = x;

    bool sus = false;

    if (sus = true) {
        sus = false;
    }

    if (z < w + 3) {
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

    3;

    x += fmadd(x, y, z);

    return 0;
}
#endif

f32 forward_decl(uint x, uint y);

struct Light {
    f32 pos_x;
    f32 pos_y;
};

#

#if 1

#

void main() {
    return 1;
}

f32 sus(f32 c) {
    // f32 c = fmadd(2, 3, 3);

    // int c = 3;
    // f32 v = c - 1;
    f32 v = 0;
    v = (0 + 3 * 4 * (5 / (3 + 4)));

    uint x = fmadd(1, 2, 3);
    // uint y = 0.7;

    // #if 0
    if (x == main()) {
        v += c;

        int v = 0;

        int w = 0;
    }
    // #endif

    c = v = 2;

    // v += vertex_main(5, 3 * v + 4, 4);
    // v *= forward_decl(v, v * v + v);
    v += 1 - v * 5;

    return c;
}
#endif
