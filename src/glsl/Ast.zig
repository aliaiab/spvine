//! The abstract syntax tree (AST) for glsl

source: []const u8,
source_name: []const u8,
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

    var parser = Parser.init(
        gpa,
        ast_node_arena,
        scratch_arena.allocator(),
        source,
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
        .source = source,
        .source_name = source_name,
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
    return self.sourceStringLocation(self.tokenString(token_index));
}

pub fn sourceStringLocation(self: Ast, source_string: []const u8) SourceLocation {
    var loc = SourceLocation{
        .source_name = self.source_name,
        .line = 1,
        .column = 1,
        .line_start = 0,
        .line_end = 0,
    };

    for (self.source, 0..) |c, i| {
        if (self.source[i..].ptr == source_string.ptr) {
            loc.line_end = @as(u32, @intCast(i));
            while (loc.line_end < self.source.len and self.source[loc.line_end] != '\n') {
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
    //TODO: handle multiple files
    const token_start = token_index.string_start;
    const token_end = token_index.string_start + token_index.string_length;

    return self.source[token_start..token_end];
}

pub fn nodeStringRange(
    self: Ast,
    node: NodePointer,
) SourceStringRange {
    const node_string_range: SourceStringRange = .{ .start = 0, .end = self.source.len };

    return self.nodeStringRecursive(node, node_string_range);
}

const SourceStringRange = struct {
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

            return result_range;
        },
        .expression_literal_boolean => {
            const literal_boolean = node.data(Node.LiteralBoolean);
            maybe_token = literal_boolean.token;
        },
        .expression_literal_number => {
            const literal_number = node.data(Node.ExpressionLiteralNumber);
            maybe_token = literal_number.token;
        },
        .expression_identifier => {
            const identifier = node.data(Node.Identifier);
            maybe_token = identifier.token;
        },
        .expression_unary_minus => {
            const binary_expr = node.data(Node.BinaryExpression);

            maybe_token = binary_expr.op_token;

            const rhs_range = self.nodeStringRecursive(.relativeFrom(node, binary_expr.right), parent_range);

            return .{
                .start = rhs_range.start -| 1,
                .end = rhs_range.end,
            };
        },
        .type_expr => {
            const identifier = node.data(Node.TypeExpr);
            maybe_token = identifier.token;
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
            value_assigned_to: Ast.NodeIndex,
            assignment: Ast.NodeIndex,
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
    file_index: u14,

    pub const nil: TokenIndex = .{ .file_index = 0, .tag = .invalid, .string_length = 0, .string_start = 0 };

    pub const end_of_file: TokenIndex = .{ .file_index = 0, .tag = .end_of_file, .string_length = 0, .string_start = 0 };

    pub fn fromToken(
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
            //TODO: handle multiple files
            .file_index = 0,
            .string_start = string_start,
            .string_length = string_length,
            .tag = token.tag,
        };
    }
};

pub const NodeIndex = packed struct(u32) {
    tag: Node.Tag,
    index: IndexInt,

    pub const IndexInt: type = std.meta.Int(.signed, @bitSizeOf(u32) - @bitSizeOf(Node.Tag));

    pub const nil: NodeIndex = .{
        //This is undefined so we don't have to waste a bit on the nil node in Node.Tag
        .tag = @enumFromInt(0),
        .index = 0,
    };

    pub fn relativeTo(parent: NodePointer, node: NodePointer) NodeIndex {
        if (node == NodePointer.nil) {
            return .nil;
        }

        const parent_int: i64 = @intCast(@intFromPtr(parent.data_ptr));
        const value_int: i64 = @intCast(@intFromPtr(node.data_ptr));

        const relative_ptr = value_int - parent_int;

        return .{
            .tag = node.tag,
            .index = @intCast(relative_ptr),
        };
    }
};

///A fat pointer type for conveniently passing nodes around
///Not to actually be stored in memory, only for locals
pub const NodePointer = packed struct {
    tag: Node.Tag,
    data_ptr: ?[*]u8,

    pub const nil: NodePointer = .{
        .tag = undefined,
        .data_ptr = null,
    };

    pub fn relativeFrom(parent: NodePointer, node_index: NodeIndex) NodePointer {
        const offset: i64 = node_index.index;

        const base: i64 = @intCast(@intFromPtr(parent.data_ptr));

        const result: u64 = @intCast(base + offset);

        return .{
            .tag = node_index.tag,
            .data_ptr = @ptrFromInt(result),
        };
    }

    pub fn data(ptr: NodePointer, comptime T: type) *T {
        if (!std.mem.isAligned(@intFromPtr(ptr.data_ptr.?), @alignOf(T))) {
            std.debug.print("expected alignment {} but found 0x{x}\n", .{ @alignOf(T), @intFromPtr(ptr.data_ptr) });
            @panic("");
        }

        return @ptrCast(@alignCast(ptr.data_ptr.?));
    }
};

pub const Node = struct {
    pub const Tag = enum(u6) {
        type_expr,
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

    pub const StructDefinition = struct {
        name: TokenIndex,
        fields: []const NodeIndex,
    };

    pub const StructField = struct {
        type_expr: NodeIndex,
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

    pub const Procedure = struct {
        return_type: NodeIndex,
        name: TokenIndex,
        param_list: NodeIndex,
        body: NodeIndex,
    };

    pub const ParamList = struct {
        params: []const NodeIndex,
    };

    pub const ParamExpr = struct {
        type_expr: NodeIndex,
        name: TokenIndex,
        qualifier: TokenIndex,
    };

    pub const BinaryExpression = struct {
        op_token: TokenIndex,
        left: NodeIndex,
        right: NodeIndex,
    };

    pub const StatementIf = struct {
        if_token: TokenIndex,
        condition_expression: NodeIndex,
        taken_statement: NodeIndex,
        not_taken_statement: NodeIndex,
    };

    pub const StatementReturn = struct {
        return_token: TokenIndex,
        expression: NodeIndex,
    };

    pub const StatementVarInit = struct {
        identifier: TokenIndex,
        qualifier: TokenIndex,
        type_expr: NodeIndex,
        array_length_specifier: NodeIndex,
        expression: NodeIndex,
    };

    pub const StatementBlock = struct {
        statements: []const NodeIndex,
    };
};

const std = @import("std");
const Ast = @This();
const Sema = @import("Sema.zig");
const Token = @import("Tokenizer.zig").Token;
const Parser = @import("Parser.zig");
