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

#if 1
//Vertex main
void vertex_main(u32 z, u32 w, u32 k) {
    f32 x = w;
    f32 y = 0;

    x = y;
    y = x;

    if (y == 10) {
        y += (3 * (w + 11)) * 11 + 3;
    } else {
        y += 3 * z + x;
    }

    y += (3) + ((3 + (NUM + 10)) + 5) + (4 + 3030) + 3 + NUM;
    x += x * 3 + 6 * 3 + y * 10;
    x += x * 10 + y;

    // x += fmadd(x, y, z);
}
#endif