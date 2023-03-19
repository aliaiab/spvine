
pub const glsl = @import("glsl.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}) {};
    defer std.debug.assert(!gpa.deinit());

    const allocator = gpa.allocator();

    var ast = try Ast.parse(allocator, @embedFile("test.glsl"));
    defer ast.deinit(allocator);

    printAst(ast, 0, 0);

    std.debug.print("\n", .{});

    printErrors("src/test.glsl", ast);
}

fn printErrors(file_path: []const u8, ast: Ast) void {
    for (ast.errors) |error_value| {
        switch (error_value.tag) {
            .expected_token => {
                const loc = ast.tokenLocation(error_value.token);
                const found_token = ast.tokens.items(.tag)[error_value.token];

                const stderr = std.io.getStdErr().writer();

                const terminal_red = "\x1B[31m";
                const terminal_green = "\x1B[32m";

                const terminal_bold = "\x1B[1;37m";

                const color_end = "\x1B[0;39m";

                stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error{s}:" ++ terminal_bold ++ " expected '{s}', found '{s}'\n" ++ color_end, .{
                    file_path,
                    loc.line + 1,
                    loc.column,
                    terminal_red,
                    color_end,
                    error_value.data.expected_token.lexeme() orelse @tagName(error_value.data.expected_token),
                    found_token.lexeme() orelse @tagName(found_token)
                }) catch {};

                stderr.print("{s}\n", .{
                    ast.source[loc.line_start..loc.line_end],
                }) catch {};

                stderr.print(terminal_green, .{}) catch {};

                for (0..ast.tokens.items(.start)[error_value.token] - loc.line_start) |_| {
                    stderr.print("~", .{}) catch {};
                }

                stderr.print("{s}\n", .{
                    "^",
                }) catch {};

                stderr.print(color_end, .{}) catch {};
                stderr.print("\n", .{}) catch {};
            },
        }
    }
}

fn printAst(ast: Ast, node: Ast.NodeIndex, depth: u32) void {
    const node_tag = ast.nodes.items(.tag)[node];

    switch (node_tag) {
        .nil => return,
        .root => {

        }, 
        else => {},
    } 

    _ = depth;
}

test {
    std.testing.refAllDecls(@This());
}

const std = @import("std");
const Parser = glsl.Parser;
const Ast = glsl.Ast;