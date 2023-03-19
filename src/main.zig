
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
                const terminal_blue = "\x1B[34m";
                const terminal_purple = "\x1B[35m";

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

                const source_line = ast.source[loc.line_start..loc.line_end];

                var tokenizer = Tokenizer { .source = source_line, .index = 0 };

                var last_token: ?Tokenizer.Token = null; 

                while (tokenizer.next()) |token| {
                    if (last_token != null) {
                        if (last_token.?.end != token.start) {
                            _ = stderr.write(source_line[last_token.?.end..token.start]) catch unreachable;
                        }
                    }

                    switch (token.tag) {
                        .directive_define,
                        .directive_undef,
                        .directive_if,
                        .directive_ifdef,
                        .directive_ifndef,
                        .directive_else,
                        .directive_elif,
                        .directive_endif,
                        .directive_error,
                        .directive_pragma,
                        .directive_extension,
                        .directive_version,
                        .directive_line,
                        .directive_end,
                        .keyword_layout,
                        .keyword_restrict,
                        .keyword_readonly,
                        .keyword_writeonly,
                        .keyword_volatile,
                        .keyword_coherent,
                        .keyword_attribute,
                        .keyword_varying,
                        .keyword_buffer,
                        .keyword_uniform,
                        .keyword_shared,
                        .keyword_const,
                        .keyword_flat,
                        .keyword_smooth,
                        .keyword_struct,
                        .keyword_void,
                        .keyword_int,
                        .keyword_uint,
                        .keyword_float,
                        .keyword_double,
                        .keyword_bool,
                        .keyword_true,
                        .keyword_false,
                        .keyword_vec2,
                        .keyword_vec3,
                        .keyword_vec4,
                        .keyword_in,
                        .keyword_out,
                        .keyword_inout,
                        => {
                            stderr.print(terminal_blue ++ "{s}" ++ color_end, .{
                                token.lexeme() orelse source_line[token.start..token.end],
                            }) catch {};  
                        },
                        .keyword_return,
                        .keyword_discard,
                        .keyword_switch,
                        .keyword_for,
                        .keyword_do,
                        .keyword_break,
                        .keyword_continue,
                        .keyword_if,
                        .keyword_else,
                        .keyword_case,
                        .keyword_default,
                        .keyword_while,
                        => {
                            stderr.print(terminal_purple ++ "{s}" ++ color_end, .{
                                token.lexeme() orelse source_line[token.start..token.end],
                            }) catch {};  
                        },
                        .literal_number => {
                            stderr.print(terminal_green ++ "{s}" ++ color_end, .{
                                token.lexeme() orelse source_line[token.start..token.end],
                            }) catch {};  
                        },
                        else => {
                            stderr.print("{s}", .{
                                token.lexeme() orelse source_line[token.start..token.end],
                            }) catch {};
                        }
                    }
                
                    last_token = token;
                }

                if (last_token.?.end != source_line.len) {
                    _ = stderr.write(terminal_green) catch unreachable;
                    _ = stderr.write(source_line[last_token.?.end..]) catch unreachable;
                    _ = stderr.write(color_end) catch unreachable;
                }

                _ = stderr.write("\n") catch unreachable;

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
const Tokenizer = glsl.Tokenizer;