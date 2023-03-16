const std = @import("std");

const Parser = @import("Parser.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}) {};
    defer std.debug.assert(!gpa.deinit());

    const allocator = gpa.allocator();

    var parser = Parser.init(allocator, @embedFile("test.glsl"));
    defer parser.deinit();

    try parser.parse();
}

test {
    std.testing.refAllDecls(@This());
}
