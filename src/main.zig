
pub const glsl = @import("glsl.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}) {};
    defer std.debug.assert(!gpa.deinit());

    const allocator = gpa.allocator();

    var ast = try Ast.parse(allocator, @embedFile("test.glsl"));
    defer ast.deinit(allocator);

    printAst(ast, 0);

    std.debug.print("\n", .{});

    for (ast.errors) |error_value| {
        switch (error_value.tag) {
            .expected_token => {
                const loc = ast.tokenLocation(error_value.token);

                std.log.err("src/test.glsl:{}:{}: expected '{s}'", .{ 
                    loc.line,
                    loc.column,
                    error_value.data.expected_token.lexeme() orelse @tagName(error_value.data.expected_token),
                });
            },
        }        
    }
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