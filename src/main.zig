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

    var stderr_buffer: [1024]u8 = undefined;
    var stderr_writer = std.fs.File.stderr().writer(&stderr_buffer);

    const stderr = &stderr_writer.interface;
    defer {
        stderr.flush() catch @panic("Flush error");
    }

    const file = try std.fs.cwd().openFile(test_glsl_path, .{});
    defer file.close();

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
        _ = token_tag; // autofix
        _ = token_index; // autofix
        // std.log.info("token_tag: {s}, '{s}'", .{ @tagName(token_tag), ast.tokenString(@intCast(token_index)) });
    }

    if (ast.errors.len != 0) {
        glsl.error_render.printErrors(test_glsl_path, ast, null, ast.errors, stderr);

        return;
    }

    {
        std.debug.print("\nglsl.Ast:\n", .{});

        var terminated_levels: std.ArrayList(u8) = .empty;
        defer terminated_levels.deinit(allocator);

        for (ast.root_decls, 0..) |root_decl, decl_index| {
            try printAst(
                allocator,
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

    var sema: glsl.Sema = .{
        .gpa = allocator,
    };
    defer sema.deinit(allocator);

    const spirv_air, const errors = try sema.analyse(ast, allocator);
    defer allocator.free(errors);
    _ = spirv_air; // autofix

    if (errors.len != 0) {
        glsl.error_render.printErrors(test_glsl_path, ast, &sema, errors, stderr);

        return;
    }
}

fn printAst(
    gpa: std.mem.Allocator,
    ast: Ast,
    terminated_levels: *std.ArrayList(u8),
    node: Ast.NodeIndex,
    depth: u32,
    sibling_index: usize,
    sibling_count: usize,
) !void {
    const node_tag = node.tag;

    if (node == Ast.NodeIndex.nil) {
        return;
    }

    const termination_index = terminated_levels.items.len;
    const is_terminator = sibling_index == sibling_count - 1;

    if (is_terminator) {
        try terminated_levels.append(gpa, @intCast(depth));
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

    var stderr_writer = std.fs.File.stderr().writer(&.{});

    const stderr = &stderr_writer.interface;
    defer {
        stderr.flush() catch @panic("Flush error");
    }

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
                                sub_sibling_count += @intFromBool(@field(node_data, payload_field.name) != Ast.NodeIndex.nil);
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

                    const symbol_map: std.EnumMap(Ast.Node.Tag, []const u8) = .init(.{
                        .expression_binary_comma = ",",
                        .expression_binary_add = "+",
                        .expression_binary_sub = "-",
                        .expression_binary_mul = "*",
                        .expression_binary_div = "/",
                        .expression_binary_eql = "==",
                        .expression_binary_assign = "=",
                        .expression_binary_assign_add = "+=",
                        .expression_binary_assign_sub = "-=",
                        .expression_binary_assign_mul = "*=",
                        .expression_binary_assign_div = "/=",
                        .expression_binary_gt = ">",
                        .expression_binary_geql = ">=",
                        .expression_binary_lt = "<",
                        .expression_binary_leql = "<=",
                        .statement_if = "if",
                    });

                    if (symbol_map.get(tag)) |node_symbol| {
                        try stderr.print("[{s}]", .{node_symbol});
                    } else {
                        try stderr.print("{s}: ", .{@tagName(tag)});
                    }

                    inline for (std.meta.fields(@TypeOf(node_data)), 0..) |payload_field, field_index| {
                        const field_value = @field(node_data, payload_field.name);

                        switch (payload_field.type) {
                            Ast.TokenIndex => {
                                try stderr.print(payload_field.name ++ ": " ++ "{s}", .{ast.tokenString(field_value)});
                                const token_location = ast.tokenLocation(field_value);

                                try stderr.print("({s}:{}:{})", .{ token_location.source_name, token_location.line, token_location.column });
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
                                if (field_value != Ast.NodeIndex.nil) {
                                    try printAst(
                                        gpa,
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
                                        gpa,
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
