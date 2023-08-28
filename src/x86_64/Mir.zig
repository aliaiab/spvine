//! The machine intermediate representation for an x86_64 instruction stream

instructions: []Instruction,

pub fn deinit(self: *Mir, allocator: std.mem.Allocator) void {
    defer self.* = undefined;

    allocator.free(self.instructions);
}

pub const Instruction = struct {
    tag: Tag,
    operands: [4]Operand = .{ .none, .none, .none, .none },

    pub const Tag = enum {
        nop,
        mov,
        lea,
        add,
        sub,
        ret,

        //vector mul single scalar
        vmulss,
        //vector div single scalar
        vdivss,
        //vector div dot product
        vdpps,
    };
};

pub const Operand = union(enum) {
    none,
    immediate: i64,
    register: Register,
};

pub const Register = enum {
    rax,
    rbx,
    rcx,
    rdx,
    rsi,
    rdi,
    rbp,
    rsp,
    r8,
    r9,
    r10,
    r11,
    r12,
    r13,
    r14,
    r15,

    xmm0,
    xmm1,
    xmm2,
    xmm3,
    xmm4,
    xmm5,
    xmm6,
    xmm7,
    xmm8,
    xmm9,
    xmm10,
    xmm11,
    xmm12,
    xmm13,
    xmm15,
};

const std = @import("std");
const Mir = @This();
