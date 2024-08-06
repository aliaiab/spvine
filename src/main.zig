pub const glsl = @import("glsl.zig");
pub const spirv = @import("spirv.zig");
pub const x86_64 = @import("x86_64.zig");

pub fn main() !void {
    var test_glsl_path: []const u8 = "src/test.glsl";

    {
        var args = std.process.args();

        _ = args.skip();

        test_glsl_path = args.next() orelse test_glsl_path;
    }

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    // defer std.debug.assert(gpa.deinit() != .leak);

    const allocator = gpa.allocator();

    const file = try std.fs.cwd().openFile(test_glsl_path, .{});
    defer file.close();

    const file_metadata = try file.metadata();
    _ = file_metadata; // autofix

    const test_glsl = try file.readToEndAlloc(allocator, std.math.maxInt(usize));
    defer allocator.free(test_glsl);

    var ast = try Ast.parse(allocator, test_glsl, test_glsl_path);
    defer ast.deinit(allocator);

    var defines = ast.defines.valueIterator();

    while (defines.next()) |val| {
        _ = val; // autofix
        // std.log.info("def: tok_idx: {}, tok_tag: {}, str: {s}", .{ val.start_token, ast.tokens.items(.tag)[val.start_token], ast.tokenString(val.start_token) });
    }

    for (ast.tokens.items(.tag), 0..) |token_tag, token_index| {
        std.log.info("token_tag: {s}, '{s}'", .{ @tagName(token_tag), ast.tokenString(@intCast(token_index)) });
    }

    if (ast.errors.len != 0) {
        printErrors(test_glsl_path, ast);

        return;
    }

    {
        std.debug.print("\nglsl.Ast:\n", .{});

        var terminated_levels = std.ArrayList(u8).init(allocator);
        defer terminated_levels.deinit();

        for (ast.root_decls, 0..) |root_decl, decl_index| {
            try printAst(
                ast,
                &terminated_levels,
                root_decl,
                0,
                decl_index,
                ast.root_decls.len,
            );
        }

        std.debug.print("\n", .{});
    }

    const spirv_air, const errors = try glsl.Sema.analyse(ast, allocator);
    defer allocator.free(errors);
    _ = spirv_air; // autofix

    if (errors.len != 0) {
        for (errors) |error_value| {
            switch (error_value) {
                .identifier_redefined => |identifier_redefined| {
                    _ = identifier_redefined; // autofix
                    std.log.err("Identifier Redefined!", .{});
                },
            }
        }

        return;
    }
}

fn printErrors(file_path: []const u8, ast: Ast) void {
    for (ast.errors) |error_value| {
        const is_same_line = ast.tokenLocation(error_value.token -| 1).line == ast.tokenLocation(error_value.token).line;

        const loc = if (is_same_line)
            ast.tokenLocation(error_value.token)
        else
            ast.tokenLocation(error_value.token - if (error_value.tag == .expected_token) @as(u32, 1) else @as(u32, 0));

        const found_token = ast.tokens.items(.tag)[error_value.token];

        const stderr = std.io.getStdErr().writer();

        const terminal_red = "\x1B[31m";
        const terminal_green = "\x1B[32m";

        const terminal_bold = "\x1B[1;37m";

        const color_end = "\x1B[0;39m";

        //Message render
        switch (error_value.tag) {
            .invalid_token => {
                stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error{s}:" ++ terminal_bold ++ " invalid token '{s}'\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    ast.tokenString(error_value.token),
                }) catch {};
            },
            .reserved_keyword_token => {
                stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error{s}:" ++ terminal_bold ++ " reserved keyword '{s}'\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    ast.tokenString(error_value.token),
                }) catch {};
            },
            .directive_error => {
                const error_directive_end = ast.tokens.items(.end)[error_value.token];

                const error_message_to_eof = ast.source[error_directive_end..];

                const error_message = error_message_to_eof[0..std.mem.indexOfScalar(u8, error_message_to_eof, '\n').?];

                stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ "{s}\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    error_message,
                }) catch {};
            },
            .unsupported_directive => {
                stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s} " ++ terminal_bold ++ "unsupported directive '{s}'" ++ "\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    ast.tokenString(error_value.token),
                }) catch {};
            },
            .expected_token => {
                if (is_same_line) {
                    stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " expected '{s}', found '{s}'\n" ++ color_end, .{
                        file_path,
                        loc.line,
                        loc.column,
                        terminal_red,
                        color_end,
                        error_value.data.expected_token.lexeme() orelse @tagName(error_value.data.expected_token),
                        found_token.lexeme() orelse @tagName(found_token),
                    }) catch {};
                } else {
                    stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " expected '{s}'\n" ++ color_end, .{
                        file_path,
                        loc.line,
                        loc.column,
                        terminal_red,
                        color_end,
                        error_value.data.expected_token.lexeme() orelse @tagName(error_value.data.expected_token),
                    }) catch {};
                }
            },
            .unexpected_token => {
                stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " unexpected '{s}'\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    found_token.lexeme() orelse @tagName(found_token),
                }) catch {};
            },
        }

        var tokenizer = Tokenizer.init(ast.source[0 .. loc.line_end + 1]);

        tokenizer.index = loc.line_start;

        var last_token: ?Tokenizer.Token = null;

        const erroring_token_start = ast.tokens.items(.start)[error_value.token];

        //Source line render
        while (tokenizer.next()) |token| {
            if (last_token != null) {
                if (last_token.?.end != token.start) {
                    _ = stderr.write(ast.source[last_token.?.end..token.start]) catch unreachable;
                }
            } else {
                for (ast.source[loc.line_start..token.start]) |char| {
                    _ = stderr.writeByte(char) catch unreachable;
                }
            }

            if (token.tag == .directive_end) {
                continue;
            }

            if (token.start == erroring_token_start) {
                _ = stderr.write(terminal_red) catch unreachable;
            }
            defer if (token.start == erroring_token_start) {
                _ = stderr.write(color_end) catch unreachable;
            };

            try printAstToken(
                stderr,
                ast,
                token,
                token.start == erroring_token_start,
            );

            last_token = token;
        }

        if (last_token != null and last_token.?.end != loc.line_end and last_token.?.tag != .directive_end) {
            _ = stderr.write(terminal_green) catch unreachable;
            _ = stderr.write(ast.source[last_token.?.end..loc.line_end]) catch unreachable;
            _ = stderr.write(color_end) catch unreachable;
        }

        _ = stderr.write("\n") catch unreachable;

        stderr.print(terminal_green, .{}) catch {};

        const cursor_length_raw = ast.tokens.items(.start)[error_value.token] - loc.line_start;

        const cursor_length = @min(cursor_length_raw, loc.line_end - loc.line_start);

        for (0..cursor_length) |_| {
            stderr.print("~", .{}) catch {};
        }

        stderr.print(terminal_red, .{}) catch {};
        stderr.print("{s}\n", .{
            "^",
        }) catch {};

        stderr.print(color_end, .{}) catch {};

        stderr.print("\n", .{}) catch {};
    }
}

fn printAstToken(
    writer: anytype,
    ast: Ast,
    token: Tokenizer.Token,
    is_erroring: bool,
) !void {
    const terminal_green = "\x1B[32m";
    const terminal_blue = "\x1B[34m";
    const terminal_purple = "\x1B[35m";
    const terminal_yellow = "\x1B[33m";
    const terminal_cyan = "\x1B[36m";
    const terminal_white = "\x1B[37m";

    const terminal_bold = "\x1B[1;37m";
    _ = terminal_bold; // autofix
    const color_end = "\x1B[0;39m";

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
        .directive_include,
        .directive_line,
        .directive_end,
        => {
            if (!is_erroring) {
                writer.print(terminal_purple, .{}) catch {};
            }

            writer.print("{s}" ++ color_end, .{
                ast.source[token.start..token.end],
            }) catch {};
        },
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
        //TODO: maybe print reserved keywords using red to indicate their 'invalidness'?
        .reserved_keyword,
        => {
            if (!is_erroring) {
                writer.print(terminal_blue, .{}) catch {};
            }

            writer.print("{s}" ++ color_end, .{
                ast.source[token.start..token.end],
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
        .left_paren,
        .right_paren,
        => {
            if (!is_erroring) {
                writer.print(terminal_purple, .{}) catch {};
            }

            writer.print("{s}" ++ color_end, .{
                ast.source[token.start..token.end],
            }) catch {};
        },
        .left_brace,
        .right_brace,
        => {
            if (!is_erroring) {
                writer.print(terminal_yellow, .{}) catch {};
            }

            writer.print("{s}" ++ color_end, .{
                ast.source[token.start..token.end],
            }) catch {};
        },
        .literal_number => {
            if (!is_erroring) {
                writer.print(terminal_green, .{}) catch {};
            }

            writer.print("{s}" ++ color_end, .{
                ast.source[token.start..token.end],
            }) catch {};
        },
        .literal_string => {
            if (!is_erroring) {
                writer.print(terminal_cyan, .{}) catch {};
            }

            writer.print("{s}" ++ color_end, .{
                ast.source[token.start..token.end],
            }) catch {};
        },
        .identifier => {
            const string = ast.source[token.start..token.end];

            if (ast.defines.get(string)) |define| {
                const token_def_start = ast.tokens.items(.start)[define.start_token];
                _ = token_def_start; // autofix
                const first_token_tag = ast.tokens.items(.tag)[define.start_token];

                switch (first_token_tag) {
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
                    => {
                        if (!is_erroring) {
                            writer.print(terminal_blue, .{}) catch {};
                        }

                        writer.print("{s}" ++ color_end, .{
                            ast.source[token.start..token.end],
                        }) catch {};
                    },
                    else => {
                        if (!is_erroring) {
                            writer.print(terminal_white, .{}) catch {};
                        }

                        writer.print("{s}" ++ color_end, .{
                            string,
                        }) catch {};
                    },
                }
            } else {
                if (!is_erroring) {
                    writer.print(terminal_white, .{}) catch {};
                }

                writer.print("{s}" ++ color_end, .{
                    string,
                }) catch {};
            }
        },
        else => {
            writer.print("{s}", .{
                ast.source[token.start..token.end],
            }) catch {};
        },
    }
}

fn printAst(
    ast: Ast,
    terminated_levels: *std.ArrayList(u8),
    node: Ast.NodeIndex,
    depth: u32,
    sibling_index: usize,
    sibling_count: usize,
) !void {
    const node_tag = node.tag;

    if (node.isNil()) {
        return;
    }

    const termination_index = terminated_levels.items.len;
    const is_terminator = sibling_index == sibling_count - 1;

    if (is_terminator) {
        try terminated_levels.append(@intCast(depth));
    }

    defer if (is_terminator) {
        terminated_levels.items[termination_index] = 255;
    };

    switch (node_tag) {
        .param_list => {
            const list = ast.dataFromNode(node, .param_list);

            if (list.params.len == 0) {
                return;
            }
        },
        .statement_block => {
            const block = ast.dataFromNode(node, .statement_block);

            if (block.statements.len == 0) {
                return;
            }
        },
        else => {},
    }

    const stderr = std.io.getStdErr().writer();

    for (0..depth) |level| {
        const is_terminated: bool = blk: {
            for (terminated_levels.items) |terminated_depth| {
                if (terminated_depth == level) {
                    break :blk true;
                }
            }

            break :blk false;
        };

        if (is_terminated) {
            try stderr.print("  ", .{});
        } else {
            try stderr.print("{s} ", .{"│"});
        }
    }

    switch (node_tag) {
        inline else => |tag| {
            switch (tag) {
                else => {
                    @setEvalBranchQuota(100000);

                    //TODO: this might make compile times bad
                    const is_leaf: bool = blk: {
                        inline for (std.meta.fields(std.meta.TagPayload(Ast.Node.Data, tag))) |field| {
                            switch (field.type) {
                                Ast.NodeIndex,
                                []const Ast.NodeIndex,
                                => {
                                    comptime break :blk false;
                                },
                                else => {},
                            }
                        }

                        break :blk true;
                    };

                    const connecting_string = if (is_terminator) "└" else "├";

                    try stderr.print("{s}", .{connecting_string});
                    try stderr.print("{s}", .{if (is_leaf) "──" else "─┬"});

                    const node_data = ast.dataFromNode(node, tag);

                    var sub_sibling_count: usize = 0;

                    inline for (std.meta.fields(@TypeOf(node_data))) |payload_field| {
                        switch (payload_field.type) {
                            Ast.NodeIndex => {
                                sub_sibling_count += @intFromBool(!@field(node_data, payload_field.name).isNil());
                            },
                            Ast.TokenIndex,
                            Tokenizer.Token.Tag,
                            => {},
                            []const Ast.NodeIndex => {
                                sub_sibling_count += @intCast(@field(node_data, payload_field.name).len);
                            },
                            else => {
                                @compileError("Node data type not supported");
                            },
                        }
                    }

                    try stderr.print("{s}: ", .{@tagName(tag)});

                    inline for (std.meta.fields(@TypeOf(node_data)), 0..) |payload_field, field_index| {
                        const field_value = @field(node_data, payload_field.name);

                        switch (payload_field.type) {
                            Ast.TokenIndex => {
                                try stderr.print(payload_field.name ++ ": " ++ "{s}", .{ast.tokenString(field_value)});
                                const token_location = ast.tokenLocation(field_value);

                                try stderr.print("({s}:{}:{}:)", .{ token_location.source_name, token_location.line, token_location.column });
                            },
                            Tokenizer.Token.Tag => {
                                try stderr.print(payload_field.name ++ ": " ++ "{s}", .{@tagName(field_value)});
                            },
                            else => {},
                        }

                        switch (payload_field.type) {
                            Ast.TokenIndex,
                            Tokenizer.Token.Tag,
                            => {
                                if (field_index != std.meta.fields(@TypeOf(node_data)).len - 1) {
                                    try stderr.print(", ", .{});
                                }
                            },
                            else => {},
                        }
                    }

                    try stderr.print("\n", .{});

                    var sub_sibling_index: usize = 0;

                    inline for (std.meta.fields(@TypeOf(node_data))) |payload_field| {
                        const field_value = @field(node_data, payload_field.name);

                        switch (payload_field.type) {
                            Ast.NodeIndex => {
                                if (!field_value.isNil()) {
                                    try printAst(
                                        ast,
                                        terminated_levels,
                                        field_value,
                                        depth + 1,
                                        sub_sibling_index,
                                        sub_sibling_count,
                                    );
                                    sub_sibling_index += 1;
                                }
                            },
                            []const Ast.NodeIndex => {
                                for (field_value, 0..) |sub_node, array_sibling_index| {
                                    try printAst(
                                        ast,
                                        terminated_levels,
                                        sub_node,
                                        depth + 1,
                                        array_sibling_index,
                                        field_value.len,
                                    );
                                }

                                if (field_value.len != 0) {
                                    sub_sibling_index += 1;
                                }
                            },
                            else => {},
                        }
                    }
                },
            }
        },
    }
}

test {
    std.testing.refAllDecls(@This());
}

const std = @import("std");
const Parser = glsl.Parser;
const Ast = glsl.Ast;
const Tokenizer = glsl.Tokenizer;
