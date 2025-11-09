// #version 450

struct Lol {
    int a;
};

#define SEVEN 7
#define SOME_CONST_2 (6 - SEVEN)
#define SOME_CONST 1 + 3 * SOME_CONST_2 + 4

Lol hash1(uint n)
{
    // hash by Hugo Elias
    n = (n << 13) ^ n;
    // return n * (n * n * 15731 + 789221) + 1376312589;
    n = SEVEN;
}

#if 0

#define f32 float
#define u32 uint

// #define NUM 0

vec2 sussyBakka(vec2 c) {
    // return sus(c * 3) + sus(1) + sus(3.2);

    dvec2 w = dvec2(1, 2) + vec2();

    return w + c;
}

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

f32 forward_decl(uint x, uint y, uint z) {
    return 1;
}

#if 0
f32 forward_decl(uint x, uint y, uint z) {
    return 1;
}
#endif

u32 forward_decl(uint x, uint y);

u32 forward_decl(uint x, uint y) {
    return -x;
}

u32 forward_decl(uint x, uint y, uint z, uint w) {
    return x + y + z + (w + true);
}

struct Light {
    vec2 pos;
    f32 pos_x;
    f32 pos_y;
};

#

#

Light sussy(f32 c) {
    // return 1;
    uint d = 1;
    float f = 3;

    if (f > d) {
        f += 3;
    } else if (true) {
        f += 5;
        if (f < 0) {
            f *= -c + 3;
        }
    }

    // return forward_decl(d * 3 + d * (d + d), d + 3, 2);
    return forward_decl(1 + 3 * 5);
    // return 1 + 2 - (1 * (3 + 1.3) + true);
}

#if 1
void main() {}

Light sus(f32 c) {
    // f32 c = fmadd(2, 3, 3);

    Light light = Light();
    // light.pos_x = light.pos_x << 3.3;

    // (light).pos = vec3(1, 2, 3);
    // light.pos_x += true + light.pos_x;

    f32 x = light.pos_x;

    x + 3 = 0;

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

    if (x > is_bum) {
        // v += c;

        is_bum += x;

        is_bum <<= 3;

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

u32 light_compute(Light light) {
    return light.pos_x + light.pos_y * light.pos;
}

mat3x3 amongus(dmat3 m) {
    return m;
}
#endif
