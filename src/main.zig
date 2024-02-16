pub const glsl = @import("glsl.zig");
pub const spirv = @import("spirv.zig");
pub const x86_64 = @import("x86_64.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(gpa.deinit() != .leak);

    const allocator = gpa.allocator();

    const test_glsl_path = "src/test.glsl";

    const file = try std.fs.cwd().openFile(test_glsl_path, .{});
    defer file.close();

    const file_metadata = try file.metadata();

    const test_glsl = try std.os.mmap(null, file_metadata.size(), std.os.PROT.READ, .{
        .TYPE = .PRIVATE,
    }, file.handle, 0);

    var ast = try Ast.parse(allocator, test_glsl);
    defer ast.deinit(allocator);

    if (false) {
        for (0..ast.tokens.len) |token_index| {
            std.log.info("{}: token: {}", .{ token_index, ast.tokens.items(.tag)[token_index] });
        }
    }

    if (ast.errors.len == 0) {
        std.debug.print("\nglsl.Ast:\n", .{});
        try printAst(ast, 0, 0, 0, 1);
        std.debug.print("\n", .{});
    }

    printErrors("src/test.glsl", ast);

    if (true) return;

    const spirv_x86_test: []align(4) const u8 = @alignCast(@embedFile("x86_64_test.vert.spv"));

    var air = try spirv.Air.fromSpirvBytes(allocator, spirv_x86_test);
    defer air.deinit(allocator);

    std.log.info("\n\nspirv air:", .{});

    std.log.info("entry_point: {s}", .{@tagName(air.capability)});

    std.log.info("capability: {s}", .{@tagName(air.capability)});
    std.log.info("memory_model: {s}", .{@tagName(air.memory_model)});
    std.log.info("addressing_mode: {s}", .{@tagName(air.addressing_mode)});

    std.log.info("", .{});

    for (air.types, 0..) |@"type", index| {
        std.log.info("{} type: {}\n", .{
            index,
            @"type",
        });
    }

    for (air.functions, 0..) |function, function_index| {
        std.log.info("{}: function: {}..{}\n", .{
            function_index,
            function.start_block,
            function.end_block,
        });

        for (air.blocks[function.start_block..function.end_block], 0..) |block, block_index| {
            std.log.info("{}: block {}", .{ block_index, block });

            for (air.instructions[block.start_instruction..block.end_instruction], 0..) |instruction, instruction_index| {
                std.log.info("{}: {}", .{ instruction_index, instruction });
            }
        }
    }

    // var mir: x86_64.Mir = .{ .instructions = x86_64_instructions.items };
    // var mir = try x86_64.Lower.lowerFromSpirvAir(allocator, air);
    // defer mir.deinit(allocator);

    // try x86_64.asm_renderer.print(mir, std.io.getStdOut().writer());
}

fn printErrors(file_path: []const u8, ast: Ast) void {
    for (ast.errors) |error_value| {
        const loc = ast.tokenLocation(error_value.token);
        const found_token = ast.tokens.items(.tag)[error_value.token + 1];

        const stderr = std.io.getStdErr().writer();

        const terminal_red = "\x1B[31m";
        const terminal_green = "\x1B[32m";

        const terminal_bold = "\x1B[1;37m";

        const color_end = "\x1B[0;39m";

        switch (error_value.tag) {
            .directive_error => {
                const error_directive_end = ast.tokens.items(.end)[error_value.token];

                const error_message_to_eof = ast.source[error_directive_end..];

                const error_message = error_message_to_eof[0..std.mem.indexOfScalar(u8, error_message_to_eof, '\n').?];

                stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error{s}:" ++ terminal_bold ++ "{s}\n" ++ color_end, .{
                    file_path,
                    loc.line + 1,
                    loc.column,
                    terminal_red,
                    color_end,
                    error_message,
                }) catch {};
            },
            .unsupported_directive => {
                stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error{s}: " ++ terminal_bold ++ "unsupported directive '{s}'" ++ "\n" ++ color_end, .{
                    file_path,
                    loc.line + 1,
                    loc.column,
                    terminal_red,
                    color_end,
                    ast.tokenString(error_value.token),
                }) catch {};
            },
            .expected_token => {
                stderr.print(terminal_bold ++ "{s}:{}:{}: {s}error{s}:" ++ terminal_bold ++ " expected '{s}', found '{s}'\n" ++ color_end, .{
                    file_path,
                    loc.line + 1,
                    loc.column,
                    terminal_red,
                    color_end,
                    error_value.data.expected_token.lexeme() orelse @tagName(error_value.data.expected_token),
                    found_token.lexeme() orelse @tagName(found_token),
                }) catch {};
            },
        }

        const source_line = ast.source[loc.line_start..loc.line_end];

        var tokenizer = Tokenizer{ .source = source_line, .index = 0 };

        var last_token: ?Tokenizer.Token = null;

        while (tokenizer.next()) |token| {
            if (last_token != null) {
                if (last_token.?.end != token.start) {
                    _ = stderr.write(source_line[last_token.?.end..token.start]) catch unreachable;
                }
            }

            try printAstToken(stderr, ast, source_line, token);

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
                token.lexeme() orelse source_line[token.start..token.end],
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
        .left_paren,
        .right_paren,
        => {
            writer.print(terminal_purple ++ "{s}" ++ color_end, .{
                token.lexeme() orelse source_line[token.start..token.end],
            }) catch {};
        },
        .left_brace,
        .right_brace,
        => {
            writer.print(terminal_yellow ++ "{s}" ++ color_end, .{
                token.lexeme() orelse source_line[token.start..token.end],
            }) catch {};
        },
        .literal_number => {
            writer.print(terminal_green ++ "{s}" ++ color_end, .{
                token.lexeme() orelse source_line[token.start..token.end],
            }) catch {};
        },
        .literal_string => {
            writer.print(terminal_cyan ++ "{s}" ++ color_end, .{
                token.lexeme() orelse source_line[token.start..token.end],
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
                            token.lexeme() orelse source_line[token.start..token.end],
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
                token.lexeme() orelse source_line[token.start..token.end],
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
