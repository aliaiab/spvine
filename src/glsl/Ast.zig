//! The abstract syntax tree (AST) for glsl

source: []const u8,
tokens: TokenList.Slice,
nodes: NodeList.Slice,
errors: []const Error,

pub fn deinit(self: *Ast, allocator: std.mem.Allocator) void {
    defer self.* = undefined;
    defer self.tokens.deinit(allocator);
    defer self.nodes.deinit(allocator);
    defer allocator.free(self.errors);
}

pub fn parse(allocator: std.mem.Allocator, source: []const u8) !Ast {
    var token_list = TokenList {};
    defer token_list.deinit(allocator);

    var preprocessor = Preprocessor.init(allocator, source);
    defer preprocessor.deinit();

    try preprocessor.tokenize(&token_list);

    var defines = preprocessor.defines.iterator();

    while (defines.next()) |define|
    {
        std.log.info("define: {s} = .{s}", .{ define.key_ptr.*, @tagName(token_list.items(.tag)[define.value_ptr.start_token]) });
    }

    var parser = Parser.init(allocator, source, token_list.slice());
    defer parser.deinit();

    try parser.parse();

    return Ast {
        .source = source,
        .tokens = token_list.toOwnedSlice(),
        .nodes = parser.nodes.toOwnedSlice(),
        .errors = try parser.errors.toOwnedSlice(allocator),
    };
}

pub const SourceLocation = struct {
    line: u32,
    column: u32,
    line_start: u32,
    line_end: u32,
};

pub fn tokenLocation(self: Ast, token_index: TokenIndex) SourceLocation {
    var loc = SourceLocation {
        .line = 0,
        .column = 0,
        .line_start = 0,
        .line_end = 0,
    };

    const token_start = self.tokens.items(.start)[token_index];

    for (self.source, 0..) |c, i| {
        if (i == token_start) {
            loc.line_end = @intCast(u32, i);
            while (loc.line_end < self.source.len and self.source[loc.line_end] != '\n') {
                loc.line_end += 1;
            }
            return loc;
        }
        if (c == '\n') {
            loc.line += 1;
            loc.column = 0;
            loc.line_start = @intCast(u32, i) + 1;
        } else {
            loc.column += 1;
        }
    }

    return loc;
}

pub const Error = struct {
    tag: Tag,
    token: Ast.TokenIndex,
    data: union {
        none: void,
        expected_token: Token.Tag,
    } = .{ .none = {} },

    pub const Tag = enum(u8) {
        expected_token,
    };
};

pub const TokenIndex = u32;
pub const NodeIndex = u32;

pub const TokenList = std.MultiArrayList(Token);
pub const NodeList = std.MultiArrayList(struct { tag: NodeTag, data: NodeData });

///Simple node structure, could be stored more in a more compact
///Format akin to std.zig.Ast
pub const NodeData = union {
    nil: void,
    generic: struct {
        left: NodeIndex = 0,
        right: NodeIndex = 0,
    },
    proc_decl: struct {
        prototype: NodeIndex = 0,
        body: NodeIndex = 0, 
    },
    proc_prototype: struct {
        return_type_token: TokenIndex = 0,
        params: []NodeIndex = &.{},
        name_token: TokenIndex = 0, 
    },
    compound_statement: struct {
        statements: []NodeIndex = &.{},
    },
};

pub const NodeTag = enum(u8) {
    nil,
    generic,
    proc_decl,
    proc_prototype,
    compound_statement,
};

const std = @import("std");
const Ast = @This();
const Token = @import("Tokenizer.zig").Token;
const Preprocessor = @import("Preprocessor.zig");
const Parser = @import("Parser.zig");