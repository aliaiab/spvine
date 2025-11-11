//! Implements error rendering to text, for display in the terminal

///Renders errors to writer in a similar style to the zig compiler
pub fn printErrors(
    file_path: []const u8,
    ast: Ast,
    sema: ?*Sema,
    errors: []const Ast.Error,
    writer: *std.Io.Writer,
) !void {
    for (errors) |error_value| {
        var is_same_line: bool = false;
        var loc: Ast.SourceLocation = undefined;
        var found_token: Tokenizer.Token.Tag = undefined;

        var error_anchor_start: usize = 0;
        var error_anchor_end: usize = 0;

        switch (error_value.anchor) {
            .token => |error_token| {
                // const previous_token: Ast.TokenIndex = @enumFromInt(@intFromEnum(error_token) -| 1);

                // is_same_line = ast.tokenLocation(previous_token).line == ast.tokenLocation(error_token).line;
                //TODO: walk back in the source stream until you find the previous token
                is_same_line = true;

                loc = ast.tokenLocation(error_token);

                // loc = if (is_same_line)
                // ast.tokenLocation(error_token)
                // else
                // ast.tokenLocation(if (error_value.tag == .expected_token) previous_token else error_token);

                found_token = error_token.tag;

                //TODO: handle multiple files
                error_anchor_start = error_token.string_start;
                error_anchor_end = error_token.string_start + @as(u32, error_token.string_length);
            },
            .node => |error_node| {
                const node_string = ast.nodeStringRange(error_node);

                error_anchor_start = node_string.start;
                error_anchor_end = node_string.end;

                loc = ast.sourceStringLocation(ast.source[error_anchor_start..error_anchor_end]);
            },
        }

        const terminal_red = "\x1B[31m";
        const terminal_green = "\x1B[32m";

        const terminal_bold = "\x1B[1;37m";

        const color_end = "\x1B[0;39m";

        //Message render
        switch (error_value.tag) {
            .invalid_token => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error{s}:" ++ terminal_bold ++ " invalid token '{s}'\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    ast.tokenString(error_value.anchor.token),
                }) catch {};
            },
            .reserved_keyword_token => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error{s}:" ++ terminal_bold ++ " reserved keyword '{s}'\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    ast.tokenString(error_value.anchor.token),
                }) catch {};
            },
            .directive_error => {
                const error_directive_end = error_value.anchor.token.string_start + error_value.anchor.token.string_length;

                const error_message_to_eof = ast.source[error_directive_end..];

                const error_message = error_message_to_eof[0..std.mem.indexOfScalar(u8, error_message_to_eof, '\n').?];

                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ "{s}\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    error_message,
                }) catch {};
            },
            .unsupported_directive => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s} " ++ terminal_bold ++ "unsupported directive '{s}'" ++ "\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    ast.tokenString(error_value.anchor.token),
                }) catch {};
            },
            .unexpected_endif => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s} " ++ terminal_bold ++ "unexpected #endif" ++ "\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                }) catch {};
            },
            .expected_endif => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s} " ++ terminal_bold ++ "Expected corresponding #endif" ++ "\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                }) catch {};
            },
            .expected_token => {
                if (is_same_line) {
                    writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " expected '{s}', found '{s}'\n" ++ color_end, .{
                        file_path,
                        loc.line,
                        loc.column,
                        terminal_red,
                        color_end,
                        error_value.data.expected_token.lexeme() orelse @tagName(error_value.data.expected_token),
                        found_token.lexeme() orelse @tagName(found_token),
                    }) catch {};
                } else {
                    writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " expected '{s}'\n" ++ color_end, .{
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
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " unexpected '{s}'\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    found_token.lexeme() orelse @tagName(found_token),
                }) catch {};
            },
            .undeclared_identifier => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " Undeclared identifier '{s}'\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    ast.tokenString(error_value.anchor.token),
                }) catch {};
            },
            .identifier_redefined => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " Identifier '{s}' redefined\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    ast.tokenString(error_value.anchor.token),
                }) catch {};
            },
            .type_mismatch => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " Type mismatch: cannot convert from type ", .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                }) catch {};

                writer.writeAll("'") catch {};
                sema.?.printTypeName(ast, writer, error_value.data.type_mismatch.rhs_type) catch {};
                writer.writeAll("'") catch {};

                writer.writeAll(" to ") catch {};

                writer.writeAll("'") catch {};
                sema.?.printTypeName(ast, writer, error_value.data.type_mismatch.lhs_type) catch {};
                writer.writeAll("'") catch {};

                writer.writeAll(color_end) catch {};
                writer.print("\n", .{}) catch {};
            },
            .type_incompatibility => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " incompatible types: ", .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                }) catch {};

                writer.writeAll("'") catch {};
                sema.?.printTypeName(ast, writer, error_value.data.type_mismatch.lhs_type) catch {};
                writer.writeAll("'") catch {};

                writer.writeAll(" and ") catch {};

                writer.writeAll("'") catch {};
                sema.?.printTypeName(ast, writer, error_value.data.type_mismatch.rhs_type) catch {};
                writer.writeAll("'") catch {};

                writer.writeAll(color_end) catch {};
                writer.print("\n", .{}) catch {};
            },
            .modified_const => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " cannot assign to a constant\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                }) catch {};
            },
            .argument_count_mismatch => {
                const argument_count_mismatch = error_value.data.argument_count_mismatch;

                if (argument_count_mismatch.actual_argument_count < argument_count_mismatch.expected_argument_count) {
                    writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " too few arguments: expected {}, found {}\n" ++ color_end, .{
                        file_path,
                        loc.line,
                        loc.column,
                        terminal_red,
                        color_end,
                        argument_count_mismatch.expected_argument_count,
                        argument_count_mismatch.actual_argument_count,
                    }) catch {};
                } else {
                    writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " too many arguments: expected {}, found {}\n" ++ color_end, .{
                        file_path,
                        loc.line,
                        loc.column,
                        terminal_red,
                        color_end,
                        argument_count_mismatch.expected_argument_count,
                        argument_count_mismatch.actual_argument_count,
                    }) catch {};
                }
            },
            .argument_count_out_of_range => {
                const argument_count_out_of_range = error_value.data.argument_count_out_of_range;

                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                }) catch {};

                if (argument_count_out_of_range.actual_argument_count < argument_count_out_of_range.expected_min_count) {
                    writer.print(" too few arguments: ", .{}) catch {};
                } else {
                    writer.print(" too many arguments: ", .{}) catch {};
                }

                if (argument_count_out_of_range.expected_min_count == argument_count_out_of_range.expected_max_count) {
                    writer.print("expected {}, found {}\n" ++ color_end, .{
                        argument_count_out_of_range.expected_min_count,
                        argument_count_out_of_range.actual_argument_count,
                    }) catch {};
                } else {
                    if (argument_count_out_of_range.actual_argument_count < argument_count_out_of_range.expected_min_count) {
                        writer.print("expected at least {}, found {}\n" ++ color_end, .{
                            argument_count_out_of_range.expected_min_count,
                            argument_count_out_of_range.actual_argument_count,
                        }) catch {};
                    } else {
                        writer.print("expected at most {}, found {}\n" ++ color_end, .{
                            argument_count_out_of_range.expected_max_count,
                            argument_count_out_of_range.actual_argument_count,
                        }) catch {};
                    }
                }
            },
            .no_matching_overload => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " no matching function overload found\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                }) catch {};
            },
            .cannot_perform_field_access => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " cannot perform field access on type ", .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                }) catch {};

                try writer.writeAll("'");
                try sema.?.printTypeName(ast, writer, error_value.data.cannot_perform_field_access.type_index);
                try writer.writeAll("'");

                try writer.print(color_end ++ "\n", .{});
            },
            .expression_not_indexable => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " expression not indexable\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                }) catch {};
            },
            .array_access_out_of_bounds => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " array index '{}' out of bounds (must be between 0 and {})\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    error_value.data.array_index_out_of_bounds.index,
                    error_value.data.array_index_out_of_bounds.array_length,
                }) catch {};
            },
            .expected_constant_expression => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " expected constant expression\n" ++ color_end, .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                }) catch {};
            },
            .no_field_in_struct => {
                writer.print(terminal_bold ++ "{s}:{}:{}: {s}error:{s}" ++ terminal_bold ++ " no field '{s}' in struct ", .{
                    file_path,
                    loc.line,
                    loc.column,
                    terminal_red,
                    color_end,
                    ast.tokenString(error_value.anchor.token),
                }) catch {};

                try writer.writeAll("'");
                try sema.?.printTypeName(ast, writer, error_value.data.no_field_in_struct.struct_type);
                try writer.writeAll("'");

                try writer.print(color_end ++ "\n", .{});
            },
        }

        var tokenizer = Tokenizer.init(ast.source[0 .. loc.line_end + 1]);

        tokenizer.index = loc.line_start;

        var last_token: ?Tokenizer.Token = null;

        //Source line render
        while (tokenizer.next()) |token| {
            if (last_token != null) {
                if (last_token.?.end != token.start) {
                    _ = writer.write(ast.source[last_token.?.end..token.start]) catch unreachable;
                }
            } else {
                for (ast.source[loc.line_start..token.start]) |char| {
                    _ = writer.writeByte(char) catch unreachable;
                }
            }

            if (token.tag == .directive_end) {
                continue;
            }

            try printAstToken(
                writer,
                ast,
                token,
            );

            last_token = token;
        }

        if (last_token != null and last_token.?.end != loc.line_end and last_token.?.tag != .directive_end) {
            _ = writer.writeAll(terminal_green) catch unreachable;
            _ = writer.writeAll(ast.source[last_token.?.end..loc.line_end]) catch unreachable;
            _ = writer.writeAll(color_end) catch unreachable;
        }

        _ = writer.writeAll("\n") catch unreachable;

        const cursor_start = error_anchor_start - loc.line_start;

        _ = writer.splatByte(' ', cursor_start) catch unreachable;

        const cursor_length = error_anchor_end - error_anchor_start;

        writer.print(terminal_red, .{}) catch {};
        writer.print("{s}", .{
            "^",
        }) catch {};

        for (0..cursor_length -| 1) |_| {
            writer.print("~", .{}) catch {};
        }

        writer.print(color_end, .{}) catch {};

        writer.print("\n", .{}) catch {};
    }
}

fn printAstToken(
    writer: *std.Io.Writer,
    ast: Ast,
    token: Tokenizer.Token,
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
            writer.print(terminal_purple, .{}) catch {};

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
            writer.print(terminal_blue, .{}) catch {};

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
            writer.print(terminal_purple, .{}) catch {};

            writer.print("{s}" ++ color_end, .{
                ast.source[token.start..token.end],
            }) catch {};
        },
        .left_brace,
        .right_brace,
        => {
            writer.print(terminal_yellow, .{}) catch {};

            writer.print("{s}" ++ color_end, .{
                ast.source[token.start..token.end],
            }) catch {};
        },
        .literal_number => {
            writer.print(terminal_green, .{}) catch {};

            writer.print("{s}" ++ color_end, .{
                ast.source[token.start..token.end],
            }) catch {};
        },
        .literal_string => {
            writer.print(terminal_cyan, .{}) catch {};

            writer.print("{s}" ++ color_end, .{
                ast.source[token.start..token.end],
            }) catch {};
        },
        .identifier => {
            const string = ast.source[token.start..token.end];

            if (true) {
                writer.print(terminal_white, .{}) catch {};

                writer.print("{s}" ++ color_end, .{
                    string,
                }) catch {};
            } else { //TODO: handle macro expansion
                if (ast.defines.get(string)) |define| {
                    const token_def_start = define.source_range.start;
                    _ = token_def_start; // autofix
                    const first_token_tag = define.start_token.tag;

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
                            writer.print(terminal_blue, .{}) catch {};

                            writer.print("{s}" ++ color_end, .{
                                ast.source[token.start..token.end],
                            }) catch {};
                        },
                        else => {
                            writer.print(terminal_white, .{}) catch {};

                            writer.print("{s}" ++ color_end, .{
                                string,
                            }) catch {};
                        },
                    }
                } else {
                    writer.print(terminal_white, .{}) catch {};

                    writer.print("{s}" ++ color_end, .{
                        string,
                    }) catch {};
                }
            }
        },
        else => {
            writer.print("{s}", .{
                ast.source[token.start..token.end],
            }) catch {};
        },
    }
}

const std = @import("std");
const Ast = @import("Ast.zig");
const Sema = @import("Sema.zig");
const Tokenizer = @import("Tokenizer.zig");
