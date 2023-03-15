const std = @import("std");

const Tokenizer = @import("Tokenizer.zig");

pub fn main() !void {
    var tokenizer = Tokenizer { .source = @embedFile("test.glsl"), .index = 0 };

    while (tokenizer.next()) |token|
    {
        std.log.info("{s} - {}:{}", .{ token.lexeme() orelse @tagName(token.tag), token.start, token.end });
    }
}

test {
    std.testing.refAllDecls(@This());
}
