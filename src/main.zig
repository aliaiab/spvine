
pub const glsl = @import("glsl.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}) {};
    defer std.debug.assert(!gpa.deinit());

    const allocator = gpa.allocator();

    var parser = Parser.init(allocator, @embedFile("test.glsl"));
    defer parser.deinit();

    try parser.parse();

    printAst(parser.ast, 0);
}

fn printAst(ast: Ast, depth: u32) void {
    for (ast.nodes.items(.tag)) |tag| {
        // if (tag == .nil) continue;

        for (0..depth) |_| {
            std.log.info("  ", .{});
        }

        std.log.info("node: {}", .{ tag });
    } 
}

test {
    std.testing.refAllDecls(@This());
}

const std = @import("std");
const Parser = glsl.Parser;
const Ast = glsl.Ast;