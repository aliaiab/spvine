#version 450

struct Lol {
    int a;
};

#define SEVEN 7
#define SOME_CONST_2 (SEVEN + 6)
#define SOME_CONST 1 + 3 * SOME_CONST_2 + 4

#define float32_t float
#define uint32_t uint

#define f32 float32_t
#define u32 uint32_t

#if 1
#include "test2.glsl"
#endif

#ifdef TEST_CONST
// #error oops
#endif

#if 0
#define gunk uint
#endif

gunk gunkFunc(uint x, uint y) {
    return x + y;
}

struct TestStruct {
    uint x;
    gunk y;
};
#if 1
u32 hash1(u32 n) {
    // n = (n << 13) ^ n;
    n = TEST_CONST + 1;
    gunk f;
    f = 3;
    f[3] = 3;
    u32 k[1 + 3 * 2] = 3;

    TestStruct funk = 3;
    if (funk) {
        funk = false;
        
        return true;
    }  

    funk.x = false;
    funk.y = true;
    funk.z = 3;
    funk.x = 3 + funk.z;

    k[5] = true;
    k[10] = 3;
    return true + n * (n * n * 15731 + 789221) + 1376312589;
}
#endif
#if 0 

float sdCone(vec3 p, vec2 c, float h) {
    // c is the sin/cos of the angle, h is height
    // Alternatively pass q instead of (c,h),
    // which is the point at the base in 2D
    vec2 q = h*vec2(c.x/c.y,-1.0);

    vec2 w = vec2( length(p.xz), p.y );
    vec2 a = w - q*clamp(dot(w,q)/dot(q,q), 0.0, 1.0 );
    vec2 b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );
    float k = sign( q.y );
    float d = min(dot( a, a ),dot(b, b));
    float s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
    return sqrt(d)*sign(s);
}
#endif

