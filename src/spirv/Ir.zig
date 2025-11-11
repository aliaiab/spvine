//! A sea of nodes representation for spirv semantics

node_buffer: std.ArrayList(u32) = .{},
effect_nodes: std.ArrayList(Node) = .{},
node_list: std.ArrayList(Node) = .{},

pub fn appendNode(
    ir: *Ir,
    allocator: std.mem.Allocator,
    node_tag: Node.Tag,
    T: type,
    node_data: T,
) !Node {
    const node_offset: usize = ir.node_buffer.items.len;

    const words_to_append = std.math.divCeil(comptime_int, @sizeOf(T), @sizeOf(u32)) catch @compileError("Type T have size divisible by 4 bytes");

    try ir.node_buffer.appendNTimes(allocator, undefined, words_to_append);

    const node: Node = .{ .offset = @intCast(node_offset) };

    ir.nodeData(node, T).* = node_data;

    switch (node_tag) {
        .return_value => {
            try ir.effect_nodes.append(allocator, node);
        },
        else => {},
    }

    return node;
}

pub fn nodeTag(ir: Ir, node: Node) *Node.Tag {
    const ptr: [*]u8 = @ptrCast(ir.node_buffer.items.ptr + node.offset);

    return @ptrCast(@alignCast(ptr));
}

pub fn nodeData(ir: Ir, node: Node, comptime T: type) *T {
    const ptr: [*]u8 = @ptrCast(ir.node_buffer.items.ptr + node.offset);

    return @ptrCast(@alignCast(ptr));
}

pub fn computeGlobalOrdering(
    ir: Ir,
    ordering: *std.ArrayList(Node),
    allocator: std.mem.Allocator,
) !void {
    for (ir.effect_nodes.items) |effect_node| {
        try ir.computeGlobalOrderingForNode(effect_node, ordering, allocator, .{ .emit_effects = true });
    }
}

///Performs global code motion to determine structure total ordering
pub fn computeGlobalOrderingForNode(
    ir: Ir,
    root_node: Node,
    ordering: *std.ArrayList(Node),
    allocator: std.mem.Allocator,
    context: struct {
        emit_effects: bool = false,
    },
) !void {
    if (context.emit_effects) {
        switch (ir.nodeTag(root_node).*) {
            .return_value => {
                const return_value = ir.nodeData(root_node, Node.ReturnValue);

                try ir.computeGlobalOrderingForNode(return_value.return_value, ordering, allocator, .{});

                try ordering.append(allocator, root_node);
            },
            else => {},
        }
    }

    switch (ir.nodeTag(root_node).*) {
        .constant => {
            try ordering.append(allocator, root_node);
        },
        .fadd => {
            const fadd = ir.nodeData(root_node, Node.FAdd);

            try ir.computeGlobalOrderingForNode(fadd.lhs, ordering, allocator, .{});
            try ir.computeGlobalOrderingForNode(fadd.rhs, ordering, allocator, .{});

            try ordering.append(allocator, root_node);
        },
        .fmul => {
            const fadd = ir.nodeData(root_node, Node.FAdd);

            try ir.computeGlobalOrderingForNode(fadd.lhs, ordering, allocator, .{});
            try ir.computeGlobalOrderingForNode(fadd.rhs, ordering, allocator, .{});

            try ordering.append(allocator, root_node);
        },
        else => {},
    }
}

pub const Node = packed struct(u32) {
    offset: u32,

    pub const nil: Node = .{ .offset = std.math.maxInt(u32) };

    pub const Tag = enum(u32) {
        null = 0,

        type_int,
        constant,
        variable,
        //Float add
        fadd,
        fmul,
        return_value,
    };

    pub const TypeInt = struct {
        tag: Tag = .type_int,
        bits: u32,
    };

    ///A scalar constant
    pub const Constant = struct {
        tag: Tag = .constant,
        type: Node,
        value_bits: u32,
    };

    pub const Variable = struct {
        tag: Tag = .variable,
        type: Node,
        ///Can be null
        initializer: Node,
    };

    pub const FAdd = struct {
        tag: Tag = .fadd,
        type: Node,
        lhs: Node,
        rhs: Node,
    };

    pub const FMul = struct {
        tag: Tag = .fmul,
        type: Node,
        lhs: Node,
        rhs: Node,
    };

    pub const ReturnValue = struct {
        tag: Tag = .return_value,
        return_value: Node,
    };
};

pub fn printNodes(ir: Ir, writer: *std.Io.Writer) !void {
    for (ir.node_list.items) |node| {
        _ = try writer.splatByteAll(' ', 4);

        switch (ir.nodeTag(node).*) {
            .null => {},
            .constant => {
                const constant = ir.nodeData(node, Node.Constant);

                try writer.print("$0x{x:0>2} := op_constant: {}\n", .{ node.offset, constant.value_bits });
            },
            .fadd => {
                const fadd = ir.nodeData(node, Node.FAdd);

                try writer.print("$0x{x:0>2} := op_fadd: $0x{x}, $0x{x}\n", .{ node.offset, fadd.lhs.offset, fadd.rhs.offset });
            },
            .fmul => {
                const fmul = ir.nodeData(node, Node.FAdd);

                try writer.print("$0x{x:0>2} := op_fmul: $0x{x}, $0x{x}\n", .{ node.offset, fmul.lhs.offset, fmul.rhs.offset });
            },
            .return_value => {
                const return_value = ir.nodeData(node, Node.ReturnValue);

                try writer.print("$0x{x:0>2} := op_return: $0x{x}\n", .{ node.offset, return_value.return_value.offset });
            },
            else => {},
        }
    }
}

const std = @import("std");
const Ir = @This();
