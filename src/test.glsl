#version 450

#define f32 float 
#define u32 uint 

// #include "simple.vert.glsl"

#define SIX 6

//expanded as: 'typedef' ^'float' 'uint'
typedef f32 uint 

//Fused multiply-add
f32 fmadd(const f32 a, const out f32 b, const f32 c) { //hello from comment!
    return a * c + c;
}

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

    y += (3) + ((3 + (SIX + 10)) + 5) + (4 + 3030) + 3 + SIX;
    x += x * 3 + 6 * 3 + y * 10;
    x += x * 10 + y;

    // x += fmadd(x, y, z);
}
