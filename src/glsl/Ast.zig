//! The abstract syntax tree (AST) for glsl

nodes: std.MultiArrayList(struct { tag: NodeTag, data: NodeData }) = .{},

pub fn deinit(self: *Ast, allocator: std.mem.Allocator) void {
    defer self.* = undefined;
    defer self.nodes.deinit(allocator);
}

pub fn addNode(self: *Ast, allocator: std.mem.Allocator, tag: NodeTag) !NodeIndex {
    const index = try self.nodes.addOne(allocator);

    self.nodes.items(.tag)[index] = tag;

    return @intCast(NodeIndex, index);
}

pub fn getNode(self: Ast, node_index: NodeIndex) *NodeData {
    return &self.nodes.items[node_index];
}

pub const TokenIndex = u32;
pub const NodeIndex = u32;

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