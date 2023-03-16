const std = @import("std");

const Preprocessor = @import("Preprocessor.zig");

pub fn main() !void {
    var preprocessor = Preprocessor.init(@embedFile("test.glsl"));

    while (preprocessor.next()) |token|
    {
        std.log.info("{s} - {}:{} ({s})", .{ token.lexeme() orelse @tagName(token.tag), token.start, token.end, preprocessor.tokenizer.source[token.start..token.end] });
    }
}

test {
    std.testing.refAllDecls(@This());
}
