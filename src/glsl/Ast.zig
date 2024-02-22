//! The abstract syntax tree (AST) for glsl

source: []const u8,
defines: ExpandingTokenizer.DefineMap,
tokens: TokenList.Slice,
nodes: NodeList.Slice,
extra_data: []const u32,
errors: []const Error,

pub fn deinit(self: *Ast, allocator: std.mem.Allocator) void {
    defer self.* = undefined;
    defer self.tokens.deinit(allocator);
    defer self.nodes.deinit(allocator);
    defer allocator.free(self.errors);
    defer allocator.free(self.extra_data);
    defer self.defines.deinit(allocator);
}

pub fn parse(allocator: std.mem.Allocator, source: []const u8) !Ast {
    var token_list = TokenList{};
    defer token_list.deinit(allocator);

    var tokenizer = ExpandingTokenizer.init(allocator, source);
    errdefer tokenizer.deinit();

    var errors: std.ArrayListUnmanaged(Error) = .{};
    // errdefer errors.deinit(allocator);

    try tokenizer.tokenize(&token_list, &errors);

    var parser = Parser.init(allocator, source, token_list.slice());
    parser.errors = errors;
    defer parser.deinit();

    parser.parse() catch |e| {
        switch (e) {
            error.ExpectedToken => {},
            error.UnexpectedToken => {},
            else => return e,
        }
    };

    return Ast{
        .source = source,
        .tokens = token_list.toOwnedSlice(),
        .nodes = parser.nodes.toOwnedSlice(),
        .extra_data = try parser.extra_data.toOwnedSlice(allocator),
        .errors = try parser.errors.toOwnedSlice(allocator),
        .defines = tokenizer.defines,
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
    var loc = SourceLocation{
        .source_name = "",
        .line = 1,
        .column = 1,
        .line_start = 0,
        .line_end = 0,
    };

    const token_start = self.tokens.items(.start)[token_index];

    for (self.source, 0..) |c, i| {
        if (i == token_start) {
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
    const token_start = self.tokens.items(.start)[token_index];
    const token_end = self.tokens.items(.end)[token_index];

    return self.source[token_start..token_end];
}

pub const Error = struct {
    tag: Tag,
    token: Ast.TokenIndex,
    data: union {
        none: void,
        expected_token: Token.Tag,
    } = .{ .none = {} },

    pub const Tag = enum(u8) {
        invalid_token,
        reserved_keyword_token,
        expected_token,
        unexpected_token,
        unsupported_directive,
        directive_error,
    };
};

pub const TokenIndex = u32;

pub const TokenList = std.MultiArrayList(Token);
pub const NodeList = std.MultiArrayList(Node);

pub const NodeIndex = packed struct(u32) {
    tag: Node.Tag,
    index: IndexInt,

    pub const IndexInt = u24;

    pub const nil: NodeIndex = .{
        .tag = .nil,
        .index = 0,
    };

    pub const root: NodeIndex = .{
        .tag = .root,
        .index = 0,
    };
};

pub const Node = struct {
    data: Data,

    pub const Data = struct {
        left: u32,
        right: u32,
    };

    pub const Tag = enum(u8) {
        nil,
        root,
        type_expr,
        procedure,
        procedure_proto,
        param_list,
        param_expr,
        procedure_body,
        statement_list,
        statement,
        statement_var_init,
        statement_assign_equal,
        statement_if,
        statement_return,
        expression_literal_number,
        expression_identifier,
        expression_binary_add,
    };

    pub const ExtraData = union(Tag) {
        nil,
        root: struct {
            decls: []const NodeIndex,
        },
        type_expr: struct {
            token: TokenIndex,
        },
        procedure: struct {
            prototype: NodeIndex,
            body: NodeIndex,
        },
        procedure_proto: struct {
            return_type: NodeIndex,
            name: TokenIndex,
            param_list: NodeIndex,
        },
        param_list: struct {
            params: []const NodeIndex,
        },
        param_expr: struct {
            type_expr: NodeIndex,
            name: TokenIndex,
        },
        procedure_body: struct {
            statements: []const NodeIndex,
        },
        statement_list: struct {
            statements: []const NodeIndex,
        },
        statement: struct {},
        statement_var_init: struct {
            type_expr: NodeIndex,
            identifier: TokenIndex,
            expression: NodeIndex,
        },
        statement_assign_equal: struct {
            identifier: TokenIndex,
            expression: NodeIndex,
        },
        statement_if: struct {
            condition_expression: NodeIndex,
            taken_statement: NodeIndex,
            not_taken_statement: NodeIndex,
        },
        statement_return: struct {
            expression: NodeIndex,
        },
        expression_literal_number: struct {
            token: TokenIndex,
        },
        expression_identifier: struct {
            token: TokenIndex,
        },
        expression_binary_add: struct {
            left: NodeIndex,
            right: NodeIndex,
        },
    };
};

pub fn dataFromNode(ast: Ast, node: NodeIndex, comptime tag: Node.Tag) std.meta.TagPayload(Node.ExtraData, tag) {
    const left = ast.nodes.items(.data)[node.index].left;
    const right = ast.nodes.items(.data)[node.index].right;

    const T = std.meta.TagPayload(Node.ExtraData, tag);

    var value: T = undefined;

    inline for (std.meta.fields(T), 0..) |field, field_index| {
        switch (field.type) {
            u32 => {
                switch (std.meta.fields(T).len) {
                    0 => {},
                    1 => {
                        @field(value, field.name) = left;
                    },
                    2 => {
                        if (field_index == 0) {
                            @field(value, field.name) = left;
                        } else {
                            @field(value, field.name) = right;
                        }
                    },
                    else => {
                        const indices = ast.extra_data[left..right];

                        @field(value, field.name) = indices[field_index];
                    },
                }
            },
            NodeIndex => {
                switch (std.meta.fields(T).len) {
                    0 => {},
                    1 => {
                        @field(value, field.name) = @bitCast(left);
                    },
                    2 => {
                        if (field_index == 0) {
                            @field(value, field.name) = @bitCast(left);
                        } else {
                            @field(value, field.name) = @bitCast(right);
                        }
                    },
                    else => {
                        const indices = ast.extra_data[left..right];

                        @field(value, field.name) = @bitCast(indices[field_index]);
                    },
                }
            },
            []const u32 => {
                if (std.meta.fields(T).len > 1) {
                    @compileError("Multiple slices not yet supported");
                }

                const indices = ast.extra_data[left..right];

                @field(value, field.name) = indices;
            },
            []const NodeIndex => {
                if (std.meta.fields(T).len > 1) {
                    @compileError("Multiple slices not yet supported");
                }

                const indices = ast.extra_data[left..right];

                @field(value, field.name) = @ptrCast(indices);
            },
            else => @compileError("Type not supported"),
        }
    }

    return value;
}

const std = @import("std");
const Ast = @This();
const Token = @import("Tokenizer.zig").Token;
const ExpandingTokenizer = @import("ExpandingTokenizer.zig");
const Parser = @import("Parser.zig");
