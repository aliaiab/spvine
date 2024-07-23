//! The abstract syntax tree (AST) for glsl

source: []const u8,
defines: ExpandingTokenizer.DefineMap,
tokens: TokenList.Slice,
node_heap: NodeHeap,
errors: []const Error,
root_decls: []const NodeIndex,

pub fn deinit(self: *Ast, allocator: std.mem.Allocator) void {
    defer self.* = undefined;
    defer self.tokens.deinit(allocator);
    defer allocator.free(self.errors);
    defer self.defines.deinit(allocator);
    defer allocator.free(self.root_decls);
    defer self.node_heap.deinit(allocator);
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
        .errors = try parser.errors.toOwnedSlice(allocator),
        .defines = tokenizer.defines,
        .root_decls = parser.root_decls,
        .node_heap = parser.node_heap,
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

pub const TokenList = std.MultiArrayList(Token);

pub const TokenIndex = u32;

pub const NodeIndex = packed struct(u32) {
    tag: Node.Tag,
    index: IndexInt,

    pub const IndexInt = u24;

    pub const nil: NodeIndex = .{
        //This is undefined so we don't have to waste a bit on the nil node in Node.Tag
        .tag = undefined,
        .index = 0,
    };

    pub inline fn isNil(self: NodeIndex) bool {
        return self.index == 0;
    }
};

pub const Node = struct {
    pub const Tag = enum(u8) {
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

pub const NodeHeap = struct {
    chunks: ChunkList = .{},
    allocated_size: u32 = 0,

    pub const NodeChunk = [1024 * 32]u8;

    const ChunkList = std.SegmentedList(NodeChunk, 0);

    pub fn deinit(self: *NodeHeap, allocator: std.mem.Allocator) void {
        self.chunks.deinit(allocator);
    }

    pub fn allocateNode(
        self: *NodeHeap,
        allocator: std.mem.Allocator,
        comptime tag: Node.Tag,
    ) !NodeIndex.IndexInt {
        const NodeType = std.meta.TagPayload(Node.ExtraData, tag);

        const node_index = try self.allocBytes(allocator, @alignOf(NodeType), @sizeOf(NodeType));

        return node_index;
    }

    pub fn getPtrFromIndex(
        self: *NodeHeap,
        index: NodeIndex.IndexInt,
        comptime T: type,
        count: usize,
    ) []T {
        const chunk_index = @divTrunc(index, @sizeOf(NodeChunk));
        const chunk_offset = index - chunk_index * @sizeOf(NodeChunk);

        const chunk: *NodeChunk = self.chunks.at(chunk_index);

        const ptr = chunk[chunk_offset..][0 .. @sizeOf(T) * count];

        const elem_ptr: [*]T = @ptrCast(@alignCast(ptr.ptr));

        return elem_ptr[0..count];
    }

    pub fn allocBytes(
        self: *NodeHeap,
        allocator: std.mem.Allocator,
        alignment: usize,
        size: usize,
    ) !NodeIndex.IndexInt {
        self.allocated_size = std.mem.alignForward(u32, self.allocated_size, @intCast(alignment));

        if (self.allocated_size + size > self.chunks.len * @sizeOf(NodeChunk)) {
            self.allocated_size = @intCast(self.chunks.len * @sizeOf(NodeChunk));
            self.allocated_size = std.mem.alignForward(u32, self.allocated_size, @intCast(alignment));

            try self.chunks.append(allocator, undefined);
        }

        const offset = self.allocated_size;

        self.allocated_size += @intCast(size);

        return @intCast(offset);
    }

    pub fn allocate(
        self: *NodeHeap,
        allocator: std.mem.Allocator,
        comptime T: type,
        count: usize,
    ) ![]T {
        const index = try self.allocBytes(allocator, @alignOf(T), count * @sizeOf(T));

        return self.getPtrFromIndex(index, T, count);
    }

    pub fn allocateDupe(
        self: *NodeHeap,
        allocator: std.mem.Allocator,
        comptime T: type,
        slice: []const T,
    ) ![]T {
        const dest = try self.allocate(allocator, T, slice.len);

        @memcpy(dest, slice);

        for (slice, dest) |a, b| {
            std.debug.assert(std.meta.eql(a, b));
        }

        return dest;
    }

    pub fn initializeNode(
        self: *NodeHeap,
        comptime node_tag: Node.Tag,
        node_payload: std.meta.TagPayload(Node.ExtraData, node_tag),
        node_index: u24,
    ) void {
        self.getNodePtr(node_tag, node_index).* = node_payload;
    }

    pub fn getNodePtr(
        self: NodeHeap,
        comptime node_tag: Node.Tag,
        node_index: u24,
    ) *std.meta.TagPayload(Node.ExtraData, node_tag) {
        const Payload = std.meta.TagPayload(Node.ExtraData, node_tag);

        std.debug.assert(node_index < self.allocated_size);

        const chunk_index = @divTrunc(node_index, @sizeOf(NodeChunk));
        const chunk_offset = node_index - chunk_index * @sizeOf(NodeChunk);

        //Sneaky hack to get around the constness metaprogramming in std.SegmentedList
        const chunks: *ChunkList = @constCast(&self.chunks);

        const chunk: [*]u8 = chunks.uncheckedAt(chunk_index);

        const bytes = (chunk + chunk_offset)[0..@sizeOf(Payload)];

        return @alignCast(std.mem.bytesAsValue(Payload, bytes));
    }

    pub fn getNodePtrConst(
        self: NodeHeap,
        comptime node_tag: Node.Tag,
        node_index: u24,
    ) *const std.meta.TagPayload(Node.ExtraData, node_tag) {
        return self.getNodePtr(node_tag, node_index);
    }

    pub fn freeNode(self: *NodeHeap, node: NodeIndex) void {
        //TODO: this really isn't necessary and is mostly a rss optimization, maybe let's just not
        var payload_size: u32 = 0;

        switch (node.tag) {
            inline else => |tag| {
                payload_size = @sizeOf(std.meta.TagPayload(Node.ExtraData, tag));
            },
        }

        if (node.index == self.allocated_size - payload_size) {
            // self.allocated_size -= payload_size;
        }
    }
};

pub fn dataFromNode(ast: Ast, node: NodeIndex, comptime tag: Node.Tag) std.meta.TagPayload(Node.ExtraData, tag) {
    return ast.node_heap.getNodePtrConst(tag, node.index).*;
}

test "Node Heap" {
    var node_heap: NodeHeap = .{};

    const node_index = try node_heap.allocateNode(std.testing.allocator, .expression_binary_add);

    node_heap.getNodePtr(.expression_binary_add, node_index).* = .{
        .left = Ast.NodeIndex.nil,
        .right = Ast.NodeIndex.nil,
    };

    try std.testing.expect(node_heap.getNodePtrConst(.expression_binary_add, node_index).left.index == 0);
    try std.testing.expect(node_heap.getNodePtrConst(.expression_binary_add, node_index).right.index == 0);

    const vals: [4]u32 = .{ 1, 2, 3, 4 };

    const vals_duped = try node_heap.allocateDupe(std.testing.allocator, u32, &vals);

    try std.testing.expect(std.mem.eql(u32, vals_duped, &vals));
}

const std = @import("std");
const Ast = @This();
const Token = @import("Tokenizer.zig").Token;
const ExpandingTokenizer = @import("ExpandingTokenizer.zig");
const Parser = @import("Parser.zig");
