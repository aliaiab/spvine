module: []const spirv.WordInt,
///The opcodes start at position 5
index: usize = 5,

pub fn initFromSlice(slice: []align(4) const u8) Iterator {
    return .{
        .module = @as([*]const spirv.WordInt, @ptrCast(@alignCast(slice.ptr)))[0 .. slice.len / @sizeOf(spirv.WordInt)],
    };
}

pub const OpData = struct {
    op: spirv.Op,
    words: []const spirv.WordInt,
};

pub fn next(self: *Iterator) ?OpData {
    if (self.index >= self.module.len) {
        return null;
    }

    const start_index = self.index;
    const word = self.module[start_index];

    const instruction_opcode = @as(spirv.Op, @enumFromInt(@as(u32, @as(u16, @truncate(word)))));
    const instruction_word_count = @as(u16, @truncate(word >> 16));

    self.index += instruction_word_count;

    return .{
        .op = instruction_opcode,
        .words = self.module[start_index..][0..instruction_word_count],
    };
}

const Iterator = @This();
const std = @import("std");
const spirv = @import("../spirv.zig");
