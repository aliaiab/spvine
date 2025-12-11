//! The abstract syntax tree (AST) for glsl

sources: []const []const u8,
source_names: []const []const u8,
defines: Parser.DefineMap,
errors: []const Error,
root_decls: []const NodePointer,

pub fn deinit(self: *Ast, allocator: std.mem.Allocator) void {
    allocator.free(self.root_decls);
    self.defines.deinit(allocator);
    allocator.free(self.errors);
    self.* = undefined;
}

pub fn parse(
    gpa: std.mem.Allocator,
    ast_node_arena: std.mem.Allocator,
    source: []const u8,
    source_name: []const u8,
) !Ast {
    var errors: std.ArrayList(Error) = .{};
    errdefer errors.deinit(gpa);

    var scratch_arena: std.heap.ArenaAllocator = .init(gpa);
    defer scratch_arena.deinit();

    var parser = try Parser.init(
        gpa,
        ast_node_arena,
        scratch_arena.allocator(),
        source,
        source_name,
    );
    parser.errors = errors;
    defer parser.deinit();

    parser.parse() catch |e| {
        switch (e) {
            error.ExpectedToken => {},
            error.UnexpectedToken => {},
            error.UnexpectedEndif => {},
            error.DirectiveError => {},
            else => return e,
        }
    };

    return Ast{
        .sources = parser.sources.items,
        .source_names = parser.source_map.keys(),
        .errors = try parser.errors.toOwnedSlice(gpa),
        .defines = parser.defines,
        .root_decls = parser.root_decls,
    };
}

///Represents the location of a token in a source character stream
pub const SourceLocation = struct {
    ///The name of the source
    source_name: []const u8,
    ///Line number starting from 0
    line: u32,
    ///Column number starting from 0
    column: u32,
    ///The start of the line in the source character stream
    line_start: u32,
    ///The end of the line in the source character stream
    line_end: u32,
};

pub fn tokenLocation(self: Ast, token_index: TokenIndex) SourceLocation {
    return self.sourceStringLocation(
        token_index.file_index,
        self.tokenString(token_index),
    );
}

pub fn sourceStringLocation(
    self: Ast,
    file_index: usize,
    source_string: []const u8,
) SourceLocation {
    var loc = SourceLocation{
        .source_name = self.source_names[file_index],
        .line = 1,
        .column = 1,
        .line_start = 0,
        .line_end = 0,
    };

    const source = self.sources[file_index];

    for (source, 0..) |c, i| {
        if (source[i..].ptr == source_string.ptr) {
            loc.line_end = @as(u32, @intCast(i));
            while (loc.line_end < source.len and source[loc.line_end] != '\n') {
                loc.line_end += 1;
            }
            return loc;
        }
        if (c == '\n') {
            loc.line += 1;
            loc.column = 1;
            loc.line_start = @as(u32, @intCast(i)) + 1;
        } else {
            loc.column += 1;
        }
    }

    return loc;
}

pub fn tokenString(self: Ast, token_index: TokenIndex) []const u8 {
    const token_start = token_index.string_start;
    const token_end = token_index.string_start + token_index.string_length;

    const source = self.sources[token_index.file_index];

    return source[token_start..token_end];
}

pub fn nodeStringRange(
    self: Ast,
    node: NodePointer,
) SourceStringRange {
    const node_string_range: SourceStringRange = .{ .start = 0, .end = self.sources[0].len, .file_index = 0 };

    return self.nodeStringRecursive(node, node_string_range);
}

pub const SourceStringRange = struct {
    file_index: usize,
    start: usize,
    end: usize,
};

fn nodeStringRecursive(
    self: Ast,
    node: NodePointer,
    parent_range: SourceStringRange,
) SourceStringRange {
    if (node == Ast.NodePointer.nil) {
        return parent_range;
    }

    var maybe_token: ?TokenIndex = null;

    var result_range: SourceStringRange = parent_range;

    switch (node.tag) {
        .expression_binary_leql,
        .expression_binary_geql,
        .expression_binary_eql,
        .expression_binary_neql,
        .expression_binary_add,
        .expression_binary_sub,
        .expression_binary_div,
        .expression_binary_mul,
        .expression_binary_comma,
        .expression_binary_proc_call,
        .expression_binary_field_access,
        .expression_binary_array_access,
        .expression_binary_bitwise_xor,
        .expression_binary_bitwise_shift_left,
        .expression_binary_bitwise_shift_right,
        .expression_binary_assign_add,
        .expression_binary_assign_mul,
        .expression_binary_assign_div,
        .expression_binary_assign_sub,
        .expression_binary_assign_bitwise_shift_left,
        .expression_binary_assign_bitwise_shift_right,
        => {
            const binary_expr = node.data(Node.BinaryExpression);

            maybe_token = binary_expr.op_token;

            const lhs_range = self.nodeStringRecursive(.relativeFrom(node, binary_expr.left), parent_range);
            const rhs_range = self.nodeStringRecursive(.relativeFrom(node, binary_expr.right), parent_range);

            result_range.start = @min(lhs_range.start, rhs_range.start);
            result_range.end = @max(lhs_range.end, rhs_range.end);
            result_range.file_index = binary_expr.op_token.file_index;

            return result_range;
        },
        .expression_literal_boolean => {
            const literal_boolean = node.data(Node.LiteralBoolean);
            maybe_token = literal_boolean.token;

            result_range.file_index = literal_boolean.token.file_index;
        },
        .expression_literal_number => {
            const literal_number = node.data(Node.ExpressionLiteralNumber);
            maybe_token = literal_number.token;

            result_range.file_index = literal_number.token.file_index;
        },
        .expression_identifier => {
            const identifier = node.data(Node.Identifier);
            maybe_token = identifier.token;

            result_range.file_index = identifier.token.file_index;
        },
        .expression_unary_minus => {
            const binary_expr = node.data(Node.BinaryExpression);

            maybe_token = binary_expr.op_token;

            const rhs_range = self.nodeStringRecursive(.relativeFrom(node, binary_expr.right), parent_range);

            return .{
                .start = rhs_range.start -| 1,
                .end = rhs_range.end,
                .file_index = rhs_range.file_index,
            };
        },
        .type_expr => {
            const identifier = node.data(Node.TypeExpr);
            maybe_token = identifier.token;
            result_range.file_index = identifier.token.file_index;
        },
        else => {
            @panic(@tagName(node.tag));
        },
    }

    if (maybe_token) |token| {
        const op_token_start = token.string_start;
        const op_token_end = token.string_start + token.string_length;

        result_range.start = @max(parent_range.start, op_token_start);
        result_range.end = @min(parent_range.end, op_token_end);
    }

    return result_range;
}

pub const Error = struct {
    tag: Tag,
    ///Points to the location in the source where the error occurs
    anchor: union(enum) {
        token: TokenIndex,
        node: NodePointer,
    },
    data: union {
        none: void,
        expected_token: Token.Tag,

        identifier_redefined: struct {
            redefinition_identifier: Ast.TokenIndex,
            ///The identifier of the original definition
            definition_identifier: Ast.TokenIndex,
        },
        modified_const: struct {
            value_assigned_to: Ast.NodeRelativePointer,
            assignment: Ast.NodeRelativePointer,
        },
        type_mismatch: struct {
            lhs_type: Sema.TypeIndex,
            rhs_type: Sema.TypeIndex,
        },
        type_incompatibility: struct {
            lhs_type: Sema.TypeIndex,
            rhs_type: Sema.TypeIndex,
        },
        argument_count_mismatch: struct {
            expected_argument_count: u32,
            actual_argument_count: u32,
        },
        argument_count_out_of_range: struct {
            expected_min_count: u32,
            expected_max_count: u32,
            actual_argument_count: u32,
        },
        array_index_out_of_bounds: struct {
            array_length: u32,
            index: i32,
        },
        no_field_in_struct: struct {
            struct_type: Sema.TypeIndex,
        },
        cannot_perform_field_access: struct {
            type_index: Sema.TypeIndex,
        },
    } = .{ .none = {} },

    pub const Tag = enum(u8) {
        invalid_token,
        reserved_keyword_token,
        expected_token,
        unexpected_token,
        unsupported_directive,
        directive_error,
        unexpected_endif,
        expected_endif,

        //Semantic errors
        undeclared_identifier,
        identifier_redefined,
        modified_const,
        type_mismatch,
        type_incompatibility,
        argument_count_mismatch,
        argument_count_out_of_range,
        no_matching_overload,
        cannot_perform_field_access,
        expression_not_indexable,
        array_access_out_of_bounds,
        expected_constant_expression,
        no_field_in_struct,
    };
};

pub const TokenIndex = packed struct(u64) {
    string_start: u32,
    tag: Token.Tag,
    ///This is the spec defined limit for identifiers, which are the longest tokens
    string_length: u10,
    ///Index into the file table, specifies which included file the token is from
    file_index: u12,
    has_line_continuation: bool = false,
    is_token_pasted: bool = false,

    pub const nil: TokenIndex = .{ .file_index = 0, .tag = .invalid, .string_length = 0, .string_start = 0 };

    pub const end_of_file: TokenIndex = .{ .file_index = 0, .tag = .end_of_file, .string_length = 0, .string_start = 0 };

    pub fn fromToken(
        source_index: usize,
        root_source: []const u8,
        sub_source: []const u8,
        token: Token,
    ) TokenIndex {
        const sub_source_start: usize = @intFromPtr(sub_source.ptr) - @intFromPtr(root_source.ptr);
        const token_start = sub_source_start + token.start;
        const token_end = sub_source_start + token.end;

        const string_start: u32 = @intCast(token_start);

        const string_length: u10 = @intCast(token_end - token_start);

        return .{
            .file_index = @intCast(source_index),
            .string_start = string_start,
            .string_length = string_length,
            .tag = token.tag,
        };
    }
};

///A relative pointer used to point to nodes within the Ast
pub const NodeRelativePointer = packed struct(u32) {
    tag: Node.Tag,
    relative_ptr: RelativePtrInt,

    pub const RelativePtrInt: type = std.meta.Int(.signed, @bitSizeOf(u32) - @bitSizeOf(Node.Tag));

    pub const nil: NodeRelativePointer = .{
        //This is undefined so we don't have to waste a bit on the nil node in Node.Tag
        .tag = @enumFromInt(0),
        .relative_ptr = 0,
    };

    pub fn relativeTo(parent: NodePointer, node: NodePointer) NodeRelativePointer {
        if (node == NodePointer.nil) {
            return .nil;
        }

        const parent_int: i64 = @intCast(parent.data_ptr);
        const value_int: i64 = @intCast(node.data_ptr);

        const relative_ptr = value_int - parent_int;

        return .{
            .tag = node.tag,
            .relative_ptr = @intCast(relative_ptr),
        };
    }
};

///A fat pointer type for conveniently passing nodes around
///Not to actually be stored in memory, only for locals
pub const NodePointer = packed struct(u64) {
    tag: Node.Tag,
    _padding: u2 = 0,
    _padding1: u8 = 0,
    //Assumes that the address space is 48 bits (which is true for x86_64 linux and windows at least but I'm not sure about other platforms)
    //TODO: add a fall back version of this struct which is a ptr + tag if needed on any platforms
    data_ptr: u48,

    pub const nil: NodePointer = .{
        .tag = undefined,
        .data_ptr = 0,
    };

    pub fn relativeFrom(parent: NodePointer, node_index: NodeRelativePointer) NodePointer {
        if (node_index == NodeRelativePointer.nil) return .nil;

        const offset: i64 = node_index.relative_ptr;

        const base: i64 = @intCast(parent.data_ptr);

        const result: u64 = @intCast(base + offset);

        return .{
            .tag = node_index.tag,
            .data_ptr = @intCast(result),
        };
    }

    pub fn data(ptr: NodePointer, comptime T: type) *T {
        return @ptrFromInt(ptr.data_ptr);
    }
};

pub const Node = struct {
    pub const Tag = enum(u6) {
        type_expr,
        type_qualifier,
        variable_decl,
        procedure,
        struct_definition,
        struct_field,
        param_list,
        param_expr,
        statement_block,
        statement_var_init,
        statement_if,
        statement_return,
        expression_literal_number,
        expression_literal_boolean,
        expression_identifier,
        expression_binary_assign,
        expression_binary_assign_add,
        expression_binary_assign_sub,
        expression_binary_assign_mul,
        expression_binary_assign_div,
        expression_binary_assign_bitwise_shift_left,
        expression_binary_assign_bitwise_shift_right,
        expression_binary_add,
        expression_binary_sub,
        expression_binary_mul,
        expression_binary_div,
        expression_binary_bitwise_xor,
        expression_binary_bitwise_shift_left,
        expression_binary_bitwise_shift_right,
        ///Less than
        expression_binary_lt,
        ///Greater than
        expression_binary_gt,
        expression_binary_eql,
        expression_binary_neql,
        ///Less than equal
        expression_binary_leql,
        ///Greater than equal
        expression_binary_geql,
        expression_binary_proc_call,
        expression_binary_comma,
        expression_binary_field_access,
        expression_binary_array_access,

        expression_unary_minus,
    };

    pub const Data = union(Tag) {
        type_expr: TypeExpr,
        type_qualifier: TypeQualifier,
        variable_decl: VariableDecl,
        procedure: Procedure,
        struct_definition: StructDefinition,
        struct_field: StructField,
        param_list: ParamList,
        param_expr: ParamExpr,
        statement_block: StatementBlock,
        statement_var_init: StatementVarInit,
        statement_if: StatementIf,
        statement_return: StatementReturn,
        expression_literal_number: ExpressionLiteralNumber,
        expression_literal_boolean: LiteralBoolean,
        expression_identifier: Identifier,
        expression_binary_assign: BinaryExpression,
        expression_binary_assign_add: BinaryExpression,
        expression_binary_assign_sub: BinaryExpression,
        expression_binary_assign_mul: BinaryExpression,
        expression_binary_assign_div: BinaryExpression,
        expression_binary_assign_bitwise_shift_left: BinaryExpression,
        expression_binary_assign_bitwise_shift_right: BinaryExpression,
        expression_binary_add: BinaryExpression,
        expression_binary_sub: BinaryExpression,
        expression_binary_mul: BinaryExpression,
        expression_binary_div: BinaryExpression,
        expression_binary_bitwise_xor: BinaryExpression,
        expression_binary_bitwise_shift_left: BinaryExpression,
        expression_binary_bitwise_shift_right: BinaryExpression,
        ///Less than
        expression_binary_lt: BinaryExpression,
        ///Greater than
        expression_binary_gt: BinaryExpression,
        expression_binary_eql: BinaryExpression,
        expression_binary_neql: BinaryExpression,
        ///Less than equal
        expression_binary_leql: BinaryExpression,
        ///Greater than equal
        expression_binary_geql: BinaryExpression,
        expression_binary_proc_call: BinaryExpression,
        expression_binary_comma: BinaryExpression,
        expression_binary_field_access: BinaryExpression,
        expression_binary_array_access: BinaryExpression,
        expression_unary_minus: BinaryExpression,
    };

    pub const VariableDecl = struct {
        qualifier: NodeRelativePointer,
        type_expr: NodeRelativePointer,
        name: TokenIndex,
    };

    pub const StructDefinition = struct {
        name: TokenIndex,
        fields: []const NodeRelativePointer,
    };

    pub const StructField = struct {
        type_expr: NodeRelativePointer,
        name: TokenIndex,
    };

    pub const Identifier = struct {
        token: TokenIndex,
    };

    pub const LiteralBoolean = struct {
        token: TokenIndex,
    };

    pub const ExpressionLiteralNumber = struct {
        token: TokenIndex,
    };

    pub const TypeExpr = struct {
        token: TokenIndex,
    };

    pub const TypeQualifier = struct {
        tokens: []const TokenIndex,
    };

    pub const Procedure = struct {
        return_type: NodeRelativePointer,
        name: TokenIndex,
        param_list: NodeRelativePointer,
        body: NodeRelativePointer,
    };

    pub const ParamList = struct {
        params: []const NodeRelativePointer,
    };

    pub const ParamExpr = struct {
        type_expr: NodeRelativePointer,
        name: TokenIndex,
        qualifier: TokenIndex,
    };

    pub const BinaryExpression = struct {
        op_token: TokenIndex,
        left: NodeRelativePointer,
        right: NodeRelativePointer,
    };

    pub const StatementIf = struct {
        if_token: TokenIndex,
        condition_expression: NodeRelativePointer,
        taken_statement: NodeRelativePointer,
        not_taken_statement: NodeRelativePointer,
    };

    pub const StatementReturn = struct {
        return_token: TokenIndex,
        expression: NodeRelativePointer,
    };

    pub const StatementVarInit = struct {
        identifier: TokenIndex,
        qualifier: TokenIndex,
        type_expr: NodeRelativePointer,
        array_length_specifier: NodeRelativePointer,
        expression: NodeRelativePointer,
    };

    pub const StatementBlock = struct {
        statements: []const NodeRelativePointer,
    };
};

///Prints a nice looking tree using unicode characters for the tree links
pub fn print(
    ast: Ast,
    writer: *std.Io.Writer,
    gpa: std.mem.Allocator,
) !void {
    var terminated_levels: std.ArrayList(u8) = .empty;
    defer terminated_levels.deinit(gpa);

    try writer.print("root:\n", .{});

    for (ast.root_decls, 0..) |root_decl, decl_index| {
        try printNode(
            ast,
            writer,
            gpa,
            &terminated_levels,
            root_decl,
            0,
            decl_index,
            ast.root_decls.len,
        );
    }
}

pub fn printNode(
    ast: Ast,
    writer: *std.Io.Writer,
    gpa: std.mem.Allocator,
    terminated_levels: *std.ArrayList(u8),
    node: Ast.NodePointer,
    depth: u32,
    sibling_index: usize,
    sibling_count: usize,
) !void {
    const node_tag = node.tag;

    if (node == Ast.NodePointer.nil) {
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
            const list = node.data(Ast.Node.ParamList);

            if (list.params.len == 0) {
                return;
            }
        },
        .statement_block => {
            const block = node.data(Ast.Node.StatementBlock);

            if (block.statements.len == 0) {
                return;
            }
        },
        else => {},
    }

    try writer.print("\n", .{});

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
            try writer.print("  ", .{});
        } else {
            try writer.print("{s} ", .{"│"});
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
                                Ast.NodeRelativePointer,
                                []const Ast.NodeRelativePointer,
                                => {
                                    comptime break :blk false;
                                },
                                else => {},
                            }
                        }

                        break :blk true;
                    };

                    const connecting_string = if (is_terminator) "└" else "├";

                    try writer.print("{s}", .{connecting_string});
                    try writer.print("{s}", .{if (is_leaf) "──" else "─┬"});

                    const node_data = node.data(std.meta.TagPayload(Ast.Node.Data, tag)).*;

                    var sub_sibling_count: usize = 0;

                    inline for (std.meta.fields(@TypeOf(node_data))) |payload_field| {
                        switch (payload_field.type) {
                            Ast.NodeRelativePointer => {
                                sub_sibling_count += @intFromBool(@field(node_data, payload_field.name) != Ast.NodeRelativePointer.nil);
                            },
                            Ast.TokenIndex,
                            Token.Tag,
                            => {},
                            []const Ast.NodeRelativePointer => {
                                sub_sibling_count += @intCast(@field(node_data, payload_field.name).len);
                            },
                            []const Ast.TokenIndex => {
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
                        try writer.print("[{s}]", .{node_symbol});
                    } else {
                        try writer.print("{s}: ", .{@tagName(tag)});
                    }

                    inline for (std.meta.fields(@TypeOf(node_data)), 0..) |payload_field, field_index| {
                        const field_value = @field(node_data, payload_field.name);

                        switch (payload_field.type) {
                            Ast.TokenIndex => {
                                try writer.print(payload_field.name ++ ": " ++ "{s}", .{ast.tokenString(field_value)});
                                const token_location = ast.tokenLocation(field_value);

                                try writer.print("({s}:{}:{})", .{ token_location.source_name, token_location.line, token_location.column });
                            },
                            Token.Tag => {
                                try writer.print(payload_field.name ++ ": " ++ "{s}", .{@tagName(field_value)});
                            },
                            else => {},
                        }

                        switch (payload_field.type) {
                            Ast.TokenIndex,
                            Token.Tag,
                            => {
                                if (field_index != std.meta.fields(@TypeOf(node_data)).len - 1) {
                                    try writer.print(", ", .{});
                                }
                            },
                            else => {},
                        }
                    }

                    var sub_sibling_index: usize = 0;

                    inline for (std.meta.fields(@TypeOf(node_data))) |payload_field| {
                        const field_value = @field(node_data, payload_field.name);

                        switch (payload_field.type) {
                            Ast.NodeRelativePointer => {
                                if (field_value != Ast.NodeRelativePointer.nil) {
                                    try printNode(
                                        ast,
                                        writer,
                                        gpa,
                                        terminated_levels,
                                        .relativeFrom(node, field_value),
                                        depth + 1,
                                        sub_sibling_index,
                                        sub_sibling_count,
                                    );
                                    sub_sibling_index += 1;
                                }
                            },
                            []const Ast.NodeRelativePointer => {
                                for (field_value, 0..) |sub_node, array_sibling_index| {
                                    try printNode(
                                        ast,
                                        writer,
                                        gpa,
                                        terminated_levels,
                                        .relativeFrom(node, sub_node),
                                        depth + 1,
                                        array_sibling_index,
                                        field_value.len,
                                    );
                                }

                                if (field_value.len != 0) {
                                    sub_sibling_index += 1;
                                }
                            },
                            []const Ast.TokenIndex => {
                                for (field_value, 0..) |token, i| {
                                    try writer.print("{s}", .{ast.tokenString(token)});

                                    if (i != field_value.len - 1) {
                                        try writer.print(", ", .{});
                                    }
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

const std = @import("std");
const Ast = @This();
const Sema = @import("Sema.zig");
const Token = @import("Tokenizer.zig").Token;
const Parser = @import("Parser.zig");
