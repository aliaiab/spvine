//! The abstract syntax tree (AST) for glsl

source: []const u8,
defines: ExpandingTokenizer.DefineMap,
tokens: TokenList.Slice,
nodes: NodeList.Slice,
extra_data: []const NodeIndex,
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

    var errors: std.ArrayListUnmanaged(Error) = .{};

    try tokenizer.tokenize(&token_list, &errors);

    var defines = tokenizer.defines.iterator();

    while (defines.next()) |define| {
        std.log.info("define: {s} = .{s}", .{ define.key_ptr.*, @tagName(token_list.items(.tag)[define.value_ptr.start_token]) });
    }

    var parser = Parser.init(allocator, source, token_list.slice());
    parser.errors = errors;
    defer parser.deinit();

    try parser.parse();

    return Ast{
        .source = source,
        .tokens = token_list.toOwnedSlice(),
        .nodes = parser.nodes.toOwnedSlice(),
        .extra_data = &.{},
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
        .line = 0,
        .column = 0,
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
            loc.column = 0;
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

pub fn rootDecls(self: Ast) []const NodeIndex {
    const data = self.nodes.items(.data)[0];

    return self.extra_data[data.left..data.right];
}

pub const Error = struct {
    tag: Tag,
    token: Ast.TokenIndex,
    data: union {
        none: void,
        expected_token: Token.Tag,
    } = .{ .none = {} },

    pub const Tag = enum(u8) {
        directive_error,
        expected_token,
        unsupported_directive,
    };
};

pub const TokenIndex = u32;
pub const NodeIndex = u32;

pub const TokenList = std.MultiArrayList(Token);
pub const NodeList = std.MultiArrayList(Node);

pub const Node = struct {
    tag: Tag,
    main_token: TokenIndex,
    data: Data,

    pub const Data = struct {
        left: NodeIndex,
        right: NodeIndex,
    };

    pub const Tag = enum(u8) {
        nil,
        root,
        proc_decl,
        proc_prototype,
        compound_statement,
        type_expr,
        param_expr,
    };
};

const std = @import("std");
const Ast = @This();
const Token = @import("Tokenizer.zig").Token;
const ExpandingTokenizer = @import("ExpandingTokenizer.zig");
const Parser = @import("Parser.zig");
