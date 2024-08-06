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
        .tag = @enumFromInt(0),
        .index = 0,
    };

    pub inline fn isNil(self: NodeIndex) bool {
        return @as(u32, @bitCast(self)) == 0;
    }
};

pub const Node = struct {
    pub const Tag = enum(u8) {
        type_expr,
        procedure,
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
        expression_binary_add,
        expression_binary_sub,
        expression_binary_mul,
        expression_binary_div,
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
    };

    pub const ExtraData = union(Tag) {
        type_expr: struct {
            token: TokenIndex,
        },
        procedure: struct {
            return_type: NodeIndex,
            name: TokenIndex,
            param_list: NodeIndex,
            body: NodeIndex,
        },
        param_list: struct {
            params: []const NodeIndex,
        },
        param_expr: struct {
            type_expr: NodeIndex,
            name: TokenIndex,
            qualifier: Token.Tag,
        },
        statement_block: struct {
            statements: []const NodeIndex,
        },
        statement_var_init: struct {
            type_expr: NodeIndex,
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
        expression_literal_boolean: struct {
            token: TokenIndex,
        },
        expression_identifier: struct {
            token: TokenIndex,
        },
        expression_binary_assign: BinaryExpression,
        expression_binary_assign_add: BinaryExpression,
        expression_binary_assign_sub: BinaryExpression,
        expression_binary_assign_mul: BinaryExpression,
        expression_binary_assign_div: BinaryExpression,
        expression_binary_add: BinaryExpression,
        expression_binary_sub: BinaryExpression,
        expression_binary_mul: BinaryExpression,
        expression_binary_div: BinaryExpression,
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

        pub const BinaryExpression = struct {
            left: NodeIndex,
            right: NodeIndex,
        };
    };
};

pub const NodeHeap = struct {
    chunks: ChunkList = .{},
    allocated_size: u32 = 0,

    pub const NodeChunk = [1024 * 64]u8;

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

        std.debug.assert(std.mem.isAligned(node_index, @alignOf(NodeType)));

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
        while (true) {
            const adjust_off = std.mem.alignPointerOffset(
                @as([*]allowzero u8, @ptrFromInt(self.allocated_size)),
                alignment,
            ) orelse return error.OutOfMemory;
            const adjusted_index = self.allocated_size + adjust_off;
            const new_end_index = adjusted_index + size;

            if (new_end_index > self.chunks.len * @sizeOf(NodeChunk)) {
                try self.chunks.append(allocator, undefined);

                continue;
            }

            self.allocated_size = @intCast(new_end_index);

            return @intCast(adjusted_index);
        }
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

        const chunk: [*]u8 = chunks.at(chunk_index);

        const bytes = (chunk + chunk_offset)[0..@sizeOf(Payload)];

        if (!std.mem.isAligned(@intFromPtr(bytes), @alignOf(Payload))) {
            std.log.info("expected alignment {}, found address x{x}", .{ @alignOf(Payload), @intFromPtr(bytes) });
        }

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
