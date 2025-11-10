#version 450

struct Lol {
    int a;
};

#define SEVEN 7
#define SOME_CONST_2 (6 - SEVEN)
#define SOME_CONST 1 + 3 * SOME_CONST_2 + 4

#define float32_t float
#define uint32_t uint

#define f32 float32_t
#define u32 uint32_t

Lol hash1(f32 n) {
    #if 0
        // hash by Hugo Elias
        n = (n << 13) ^ n;
    #endif 

    #ifdef SOME_CONST

    // return n * (n * n * 15731 + 789221) + 1376312589;
    n = SOME_CONST;

    #elif
        #error SOME_CONST not defined!
    #endif

    return n;
}

#if 1

#if 0
#error oh dear....
#endif

uint vecLen(vec2 p) {
    return p.x + p.y;
}

float sdCone(vec3 p, vec2 c, float h) {
  // c is the sin/cos of the angle, h is height
  // Alternatively pass q instead of (c,h),
  // which is the point at the base in 2D
  vec2 q = h*vec2(c.x/c.y,-1.0);
    
  vec2 w = vec2( length(p.xz), p.y );
  vec2 a = w - q*clamp( dot(w,q)/dot(q,q), 0.0, 1.0 );
  vec2 b = w - q*vec2( clamp( w.x/q.x, 0.0, 1.0 ), 1.0 );
  float k = sign( q.y );
  float d = min(dot( a, a ),dot(b, b));
  float s = max( k*(w.x*q.y-w.y*q.x),k*(w.y-q.y)  );
  return sqrt(d)*sign(s);
}
// #error fuck and shit
#endif
