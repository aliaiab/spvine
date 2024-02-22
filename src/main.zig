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
    defer std.debug.assert(gpa.deinit() != .leak);

    const allocator = gpa.allocator();

    const file = try std.fs.cwd().openFile(test_glsl_path, .{});
    defer file.close();

    const file_metadata = try file.metadata();

    const test_glsl = try std.os.mmap(null, file_metadata.size(), std.os.PROT.READ, .{
        .TYPE = .PRIVATE,
    }, file.handle, 0);

    var ast = try Ast.parse(allocator, test_glsl);
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
        try printAst(ast, 0, 0, 0, 1);
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

        const loc = if (is_same_line) ast.tokenLocation(error_value.token) else ast.tokenLocation(error_value.token - if (error_value.tag == .expected_token) @as(u32, 1) else @as(u32, 0));
        const found_token = ast.tokens.items(.tag)[error_value.token];

        const stderr = std.io.getStdErr().writer();

        const terminal_red = "\x1B[31m";
        const terminal_green = "\x1B[32m";

        const terminal_bold = "\x1B[1;37m";

        const color_end = "\x1B[0;39m";

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

                stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error{s}:" ++ terminal_bold ++ "{s}\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    error_message,
                }) catch {};
            },
            .unsupported_directive => {
                stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error{s}: " ++ terminal_bold ++ "unsupported directive '{s}'" ++ "\n" ++ color_end, .{
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
                    stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error{s}:" ++ terminal_bold ++ " expected '{s}', found '{s}'\n" ++ color_end, .{
                        file_path,
                        loc.line,
                        loc.column,
                        terminal_red,
                        color_end,
                        error_value.data.expected_token.lexeme() orelse @tagName(error_value.data.expected_token),
                        found_token.lexeme() orelse @tagName(found_token),
                    }) catch {};
                } else {
                    stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error{s}:" ++ terminal_bold ++ " expected '{s}'\n" ++ color_end, .{
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
                stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error{s}:" ++ terminal_bold ++ " unexpected '{s}'\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    found_token.lexeme() orelse @tagName(found_token),
                }) catch {};
            },
        }

        const source_line = ast.source[loc.line_start .. loc.line_end + 1];

        var tokenizer = Tokenizer.init(ast.source[loc.line_start .. loc.line_end + 1]);

        var last_token: ?Tokenizer.Token = null;

        while (tokenizer.next()) |token| {
            if (last_token != null) {
                if (last_token.?.end != token.start) {
                    _ = stderr.write(source_line[last_token.?.end..token.start]) catch unreachable;
                }
            } else {
                for (source_line[0..token.start]) |char| {
                    if (char != ' ') continue;

                    _ = stderr.writeByte(char) catch unreachable;
                }
            }

            try printAstToken(stderr, ast, source_line, token);

            last_token = token;
        }

        if (last_token != null and last_token.?.end != source_line.len) {
            _ = stderr.write(terminal_green) catch unreachable;
            _ = stderr.write(source_line[last_token.?.end..]) catch unreachable;
            _ = stderr.write(color_end) catch unreachable;
        }

        if (source_line[source_line.len - 1] != '\n') {
            _ = stderr.write("\n") catch unreachable;
        }

        stderr.print(terminal_green, .{}) catch {};

        const cursor_length_raw = ast.tokens.items(.start)[error_value.token] - loc.line_start;

        const cursor_length = @min(cursor_length_raw, loc.line_end - loc.line_start);

        for (0..cursor_length) |_| {
            stderr.print("~", .{}) catch {};
        }

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
    source_line: []const u8,
    token: Tokenizer.Token,
) !void {
    const terminal_green = "\x1B[32m";
    const terminal_blue = "\x1B[34m";
    const terminal_purple = "\x1B[35m";
    const terminal_yellow = "\x1B[33m";
    const terminal_cyan = "\x1B[36m";
    const terminal_white = "\x1B[37m";
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
            writer.print(terminal_purple ++ "{s}" ++ color_end, .{
                source_line[token.start..token.end],
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
        => {
            writer.print(terminal_blue ++ "{s}" ++ color_end, .{
                source_line[token.start..token.end],
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
            writer.print(terminal_purple ++ "{s}" ++ color_end, .{
                source_line[token.start..token.end],
            }) catch {};
        },
        .left_brace,
        .right_brace,
        => {
            writer.print(terminal_yellow ++ "{s}" ++ color_end, .{
                source_line[token.start..token.end],
            }) catch {};
        },
        .literal_number => {
            writer.print(terminal_green ++ "{s}" ++ color_end, .{
                source_line[token.start..token.end],
            }) catch {};
        },
        .literal_string => {
            writer.print(terminal_cyan ++ "{s}" ++ color_end, .{
                source_line[token.start..token.end],
            }) catch {};
        },
        .identifier => {
            const string = source_line[token.start..token.end];

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
                        writer.print(terminal_blue ++ "{s}" ++ color_end, .{
                            source_line[token.start..token.end],
                        }) catch {};
                    },
                    else => {
                        writer.print(terminal_white ++ "{s}" ++ color_end, .{
                            string,
                        }) catch {};
                    },
                }
            } else {
                writer.print(terminal_white ++ "{s}" ++ color_end, .{
                    string,
                }) catch {};
            }
        },
        else => {
            writer.print("{s}", .{
                source_line[token.start..token.end],
            }) catch {};
        },
    }
}

fn printAst(
    ast: Ast,
    node: Ast.NodeIndex,
    depth: u32,
    sibling_index: usize,
    sibling_count: usize,
) !void {
    const node_tag = ast.nodes.items(.tag)[node];

    if (node == 0 and depth != 0) {
        return;
    }

    const stderr = std.io.getStdErr().writer();

    for (0..depth) |_| {
        try stderr.print("  ", .{});
    }

    const connecting_string = if (sibling_index == sibling_count - 1) "└" else "├";
    const is_leaf = switch (node_tag) {
        .type_expr => true,
        .statement => true,
        else => false,
    };

    try stderr.print("{s}", .{connecting_string});
    try stderr.print("{s}", .{if (is_leaf) "──" else "─┬"});

    switch (node_tag) {
        .nil => return,
        .root => {
            try stderr.print("root: \n", .{});

            const root_nodes = ast.extra_data[ast.nodes.items(.data)[node].left..ast.nodes.items(.data)[node].right];

            for (root_nodes, 0..) |child, child_index| {
                try printAst(ast, child, depth + 1, child_index, root_nodes.len);
            }
        },
        .procedure => {
            try stderr.print("proc_decl:\n", .{});

            const proc = ast.dataFromNode(node, .procedure);

            try printAst(ast, proc.prototype, depth + 1, 0, 2);
            try printAst(ast, proc.body, depth + 1, 1, 2);
        },
        .procedure_proto => {
            try stderr.print("proc_prototype: ", .{});

            const proto = ast.dataFromNode(node, .procedure_proto);

            try stderr.print("{s}\n", .{ast.tokenString(proto.name)});

            try printAst(ast, proto.return_type, depth + 1, 0, 2);
            try printAst(ast, proto.param_list, depth + 1, 1, 2);
        },
        .type_expr => {
            const type_expr = ast.dataFromNode(node, .type_expr);

            try stderr.print("type_expr: {s}\n", .{ast.tokenString(type_expr.token)});
        },
        .param_expr => {
            const node_data = ast.nodes.get(node);

            const type_expr = node_data.data.left;
            const param_identifier = node_data.data.right;
            try stderr.print("param_expr: {s}\n", .{ast.tokenString(param_identifier)});

            try printAst(ast, type_expr, depth + 1, 0, 1);
        },
        .param_list => {
            const list = ast.dataFromNode(node, .param_list);

            try stderr.print("param_list:\n", .{});

            for (list.params, 0..) |param_expr, child_index| {
                try printAst(
                    ast,
                    param_expr,
                    depth + 1,
                    child_index,
                    list.params.len,
                );
            }
        },
        .procedure_body => {
            const list = ast.dataFromNode(node, .procedure_body);

            try stderr.print("proc_body:\n", .{});

            for (list.statements, 0..) |statement, statement_index| {
                try printAst(ast, statement, depth + 1, statement_index, list.statements.len);
            }
        },
        .statement_list => {
            const list = ast.dataFromNode(node, .statement_list);

            try stderr.print("statement_list:\n", .{});

            for (list.statements, 0..) |statement, statement_index| {
                try printAst(ast, statement, depth + 1, statement_index, list.statements.len);
            }
        },
        .statement_var_init => {
            const var_init = ast.dataFromNode(node, .statement_var_init);

            try stderr.print("statement_var_init: {s}\n", .{ast.tokenString(var_init.identifier)});

            try printAst(ast, var_init.type_expr, depth + 1, 0, 2);
            try printAst(ast, var_init.expression, depth + 1, 1, 2);
        },
        .statement_if => {
            const if_statement = ast.dataFromNode(node, .statement_if);

            try stderr.print("statement_if: \n", .{});

            try printAst(ast, if_statement.condition_expression, depth + 1, 0, 3);
            try printAst(ast, if_statement.taken_statement, depth + 1, 1, 3);
            try printAst(ast, if_statement.not_taken_statement, depth + 1, 2, 3);
        },
        .statement_return => {
            const return_statement = ast.dataFromNode(node, .statement_return);
            try stderr.print("statement_return:\n", .{});

            try printAst(ast, return_statement.expression, depth + 1, 0, 1);
        },
        .expression_binary_add => {
            const binary_add = ast.dataFromNode(node, .expression_binary_add);

            try stderr.print("expression_binary_add:\n", .{});

            try printAst(ast, binary_add.left, depth + 1, 0, 2);
            try printAst(ast, binary_add.right, depth + 1, 1, 2);
        },
        else => {
            try stderr.print("node: .{s}:\n", .{@tagName(node_tag)});
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
