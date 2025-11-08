//! The abstract syntax tree (AST) for glsl

source: []const u8,
source_name: []const u8,
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

pub fn parse(
    allocator: std.mem.Allocator,
    source: []const u8,
    source_name: []const u8,
) !Ast {
    var token_list = TokenList{};
    defer token_list.deinit(allocator);

    var tokenizer = ExpandingTokenizer.init(allocator, source);
    errdefer tokenizer.deinit();

    var errors: std.ArrayList(Error) = .{};
    errdefer errors.deinit(allocator);

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
        .source_name = source_name,
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
    const token_start = self.tokens.items(.start)[@intFromEnum(token_index)];
    const token_end = self.tokens.items(.end)[@intFromEnum(token_index)];

    return self.source[token_start..token_end];
}

pub fn nodeStringRange(
    self: Ast,
    node: NodeIndex,
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
    node: NodeIndex,
    parent_range: SourceStringRange,
) SourceStringRange {
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
        => {
            const binary_expr: Node.BinaryExpression = self.dataFromNode(node, .expression_binary_add);

            maybe_token = binary_expr.op_token;

            const lhs_range = self.nodeStringRecursive(binary_expr.left, parent_range);
            const rhs_range = self.nodeStringRecursive(binary_expr.right, parent_range);

            result_range.start = @min(lhs_range.start, rhs_range.start);
            result_range.end = @max(lhs_range.end, rhs_range.end);

            return result_range;
        },
        .expression_literal_boolean => {
            const literal_boolean: Node.LiteralBoolean = self.dataFromNode(node, .expression_literal_boolean);
            maybe_token = literal_boolean.token;
        },
        .expression_literal_number => {
            const literal_number: Node.ExpressionLiteralNumber = self.dataFromNode(node, .expression_literal_number);
            maybe_token = literal_number.token;
        },
        .expression_identifier => {
            const identifier: Node.Identifier = self.dataFromNode(node, .expression_identifier);
            maybe_token = identifier.token;
        },
        else => {
            @panic(@tagName(node.tag));
        },
    }

    if (maybe_token) |token| {
        const op_token_start = self.tokens.items(.start)[@intFromEnum(token)];
        const op_token_end = self.tokens.items(.end)[@intFromEnum(token)];

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
        node: NodeIndex,
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
    } = .{ .none = {} },

    pub const Tag = enum(u8) {
        invalid_token,
        reserved_keyword_token,
        expected_token,
        unexpected_token,
        unsupported_directive,
        directive_error,

        //Semantic errors
        undeclared_identifier,
        identifier_redefined,
        modified_const,
        type_mismatch,
        type_incompatibility,
        argument_count_mismatch,
        argument_count_out_of_range,
        no_matching_overload,
    };
};

pub const TokenList = std.MultiArrayList(Token);

pub const TokenIndex = enum(u32) {
    _,
};

pub const NodeIndex = packed struct(u32) {
    tag: Node.Tag,
    index: IndexInt,

    pub const IndexInt = std.meta.Int(.unsigned, @bitSizeOf(u32) - @bitSizeOf(Node.Tag));

    pub const nil: NodeIndex = .{
        //This is undefined so we don't have to waste a bit on the nil node in Node.Tag
        .tag = @enumFromInt(0),
        .index = 0,
    };
};

pub const Node = struct {
    pub const Tag = enum(u5) {
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
        qualifier: Token.Tag,
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
        qualifier: Token.Tag,
        type_expr: NodeIndex,
        expression: NodeIndex,
    };

    pub const StatementBlock = struct {
        statements: []const NodeIndex,
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
        const NodeType = std.meta.TagPayload(Node.Data, tag);

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
        node_payload: std.meta.TagPayload(Node.Data, node_tag),
        node_index: u24,
    ) void {
        self.getNodePtr(node_tag, node_index).* = node_payload;
    }

    pub fn getNodePtr(
        self: NodeHeap,
        comptime node_tag: Node.Tag,
        node_index: NodeIndex.IndexInt,
    ) *std.meta.TagPayload(Node.Data, node_tag) {
        const Payload = std.meta.TagPayload(Node.Data, node_tag);

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
        node_index: NodeIndex.IndexInt,
    ) *const std.meta.TagPayload(Node.Data, node_tag) {
        return self.getNodePtr(node_tag, node_index);
    }

    pub fn freeNode(self: *NodeHeap, node: NodeIndex) void {
        //TODO: this really isn't necessary and is mostly a rss optimization, maybe let's just not
        var payload_size: u32 = 0;

        @setEvalBranchQuota(100000);

        switch (node.tag) {
            inline else => |tag| {
                payload_size = @sizeOf(std.meta.TagPayload(Node.Data, tag));
            },
        }

        if (node.index == self.allocated_size - payload_size) {
            // self.allocated_size -= payload_size;
        }
    }
};

pub fn dataFromNode(ast: Ast, node: NodeIndex, comptime tag: Node.Tag) std.meta.TagPayload(Node.Data, tag) {
    return ast.node_heap.getNodePtrConst(tag, node.index).*;
}

test "Node Heap" {
    var node_heap: NodeHeap = .{};

    const node_index = try node_heap.allocateNode(std.testing.allocator, .expression_binary_add);

    node_heap.getNodePtr(.expression_binary_add, node_index).* = .{
        .op_token = 0,
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
const Sema = @import("Sema.zig");
const Token = @import("Tokenizer.zig").Token;
const ExpandingTokenizer = @import("ExpandingTokenizer.zig");
const Parser = @import("Parser.zig");
