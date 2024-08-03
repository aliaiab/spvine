#version 450

#  define f32 float 
#  define u32 uint 

//#include "simple.vert.glsl"

#define SIX 6

f32 add(f32 a, f32 b) {
    return a + b * 10;
}

//Vertex main
void vertex_main(
    u32 z,
    u32 w,
    u32 k,
) {
    f32 x = w;
    f32 y = 0;

    x = y;
    y = x;

    if (y + 10) {
        y += 3 * w;
    }

    y += (3) + ((3 + (SIX + 10)) + 5) + (4 + 3030) + 3 + SIX;
    x += x + y * 10;
    x += x * 10 + y;
}