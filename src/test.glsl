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
u32 fmadd(const f32 a, const f32 b, const f32 c) { //hello from comment!
    return (a * c + c);
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

f32 forward_decl(uint x, uint y) {
    return 1;
}

struct Light {
    f32 pos_x;
    f32 pos_y;
};

#

#

Light sussy() {
    // return 1;
    // return forward_decl(1, true);
    return 1 + 2 - (1 * (3 + 1.3) + true);
}

#if 1
void main() {}

Light sus(f32 c) {
    // f32 c = fmadd(2, 3, 3);

    // main f = 0;

    // int c = 3;
    // f32 v = c - 1;
    // u32 w = 1.3;
    u32 v = 0;
    // v += 0.3;
    v = (0 + 3 * 4 * (5 / (3 + 4)));

    // uint x = fmadd(1, 2, 3);
    int x = 0;
    // uint y = 0.7;

    // #if 0
    uint is_bum = 0;

    if (x == 1) {
        // v += c;

        is_bum += x;

        int v = 0;

        int w = 0;
    }
    // #endif

    // c = v = 2;

    // v += vertex_main(5, 3 * v + 4, 4);
    // v *= forward_decl(v, v * v + v);
    v += 1 - v * 5;

    return c * 3 + 2;
}
#endif
