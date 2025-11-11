//! A sea of nodes representation for spirv semantics

node_buffer: std.ArrayList(u32) = .{},
effect_nodes: std.ArrayList(Node) = .{},
node_list: std.ArrayList(Node) = .{},
types: std.AutoArrayHashMapUnmanaged(Node, void) = .{},
constants: std.AutoArrayHashMapUnmanaged(Node.Constant, Node) = .{},

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
    ir: *Ir,
    ordering: *std.ArrayList(Node),
    allocator: std.mem.Allocator,
) !void {
    for (ir.effect_nodes.items) |effect_node| {
        try ir.computeGlobalOrderingForNode(effect_node, ordering, allocator, .{ .emit_effects = true });
    }
}

///Performs global code motion to determine structure total ordering
pub fn computeGlobalOrderingForNode(
    ir: *Ir,
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
        //Side effecting functions musn't be revaluated
        .null,
        .return_value,
        => {},
        .type_int,
        .type_float,
        => {
            _ = try ir.deduplicateTypeOrConstant(allocator, root_node);
        },
        .variable => {},
        .constant => {
            _ = try ir.deduplicateTypeOrConstant(allocator, root_node);
        },
        .fadd => {
            const fadd = ir.nodeData(root_node, Node.FAdd);

            fadd.type = try ir.deduplicateTypeOrConstant(allocator, fadd.type);
            fadd.lhs = try ir.deduplicateTypeOrConstant(allocator, fadd.lhs);
            fadd.rhs = try ir.deduplicateTypeOrConstant(allocator, fadd.rhs);

            try ir.computeGlobalOrderingForNode(fadd.lhs, ordering, allocator, .{});
            try ir.computeGlobalOrderingForNode(fadd.rhs, ordering, allocator, .{});

            try ordering.append(allocator, root_node);
        },
        .fsub => {
            const fsub = ir.nodeData(root_node, Node.FSub);

            fsub.type = try ir.deduplicateTypeOrConstant(allocator, fsub.type);
            fsub.lhs = try ir.deduplicateTypeOrConstant(allocator, fsub.lhs);
            fsub.rhs = try ir.deduplicateTypeOrConstant(allocator, fsub.rhs);

            try ir.computeGlobalOrderingForNode(fsub.lhs, ordering, allocator, .{});
            try ir.computeGlobalOrderingForNode(fsub.rhs, ordering, allocator, .{});

            try ordering.append(allocator, root_node);
        },
        .fmul => {
            const fmul = ir.nodeData(root_node, Node.FAdd);

            fmul.type = try ir.deduplicateTypeOrConstant(allocator, fmul.type);
            fmul.lhs = try ir.deduplicateTypeOrConstant(allocator, fmul.lhs);
            fmul.rhs = try ir.deduplicateTypeOrConstant(allocator, fmul.rhs);

            try ir.computeGlobalOrderingForNode(fmul.lhs, ordering, allocator, .{});
            try ir.computeGlobalOrderingForNode(fmul.rhs, ordering, allocator, .{});

            try ordering.append(allocator, root_node);
        },
        .fnegate => {
            const fnegate = ir.nodeData(root_node, Node.FNegate);

            fnegate.type = try ir.deduplicateTypeOrConstant(allocator, fnegate.type);
            fnegate.operand = try ir.deduplicateTypeOrConstant(allocator, fnegate.operand);

            try ir.computeGlobalOrderingForNode(fnegate.operand, ordering, allocator, .{});

            try ordering.append(allocator, root_node);
        },
    }
}

pub fn deduplicateTypeOrConstant(ir: *Ir, allocator: std.mem.Allocator, node: Node) !Node {
    switch (ir.nodeTag(node).*) {
        .type_int,
        .type_float,
        => {
            if (ir.types.get(node)) |_| {
                return node;
            }

            try ir.types.put(allocator, node, {});

            return node;
        },
        .constant => {
            const constant = ir.nodeData(node, Node.Constant);

            constant.type = try ir.deduplicateTypeOrConstant(allocator, constant.type);

            if (ir.constants.get(constant.*)) |new_node| {
                return new_node;
            }

            try ir.constants.put(allocator, constant.*, node);

            return node;
        },
        else => return node,
    }
}

pub const Node = packed struct(u32) {
    offset: u32,

    pub const nil: Node = .{ .offset = std.math.maxInt(u32) };

    pub const Tag = enum(u32) {
        null = 0,

        type_int,
        type_float,

        constant,
        variable,
        //Float add
        fadd,
        fsub,
        fmul,
        fnegate,
        return_value,
    };

    pub const TypeFloat = struct {
        tag: Tag = .type_float,
        bit_width: u32,
    };

    pub const TypeInt = struct {
        tag: Tag = .type_int,
        bit_width: u16,
        signedness: Signedness,
        range_value_min: u32,
        range_value_max: u32,

        pub const Signedness = enum(u16) {
            unsigned,
            signed,
        };
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

    pub const FSub = struct {
        tag: Tag = .fsub,
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

    pub const FNegate = struct {
        tag: Tag = .fnegate,
        type: Node,
        operand: Node,
    };

    pub const ReturnValue = struct {
        tag: Tag = .return_value,
        return_value: Node,
    };
};

pub fn printNodes(ir: Ir, writer: *std.Io.Writer) !void {
    _ = try writer.print("Types:\n\n", .{});

    for (ir.types.keys()) |node| {
        _ = try writer.splatByteAll(' ', 4);

        switch (ir.nodeTag(node).*) {
            .null => {},
            .type_int => {
                const type_int = ir.nodeData(node, Node.TypeInt);

                try writer.print("$0x{x:0>2} := type_int: bit_width: {}\n", .{ node.offset, type_int.bit_width });
            },
            .type_float => {
                const type_float = ir.nodeData(node, Node.TypeFloat);

                try writer.print("$0x{x:0>2} := type_float: bit_width: {}\n", .{ node.offset, type_float.bit_width });
            },
            else => {},
        }
    }

    _ = try writer.print("\nConstants:\n\n", .{});

    for (ir.constants.values()) |node| {
        _ = try writer.splatByteAll(' ', 4);

        switch (ir.nodeTag(node).*) {
            .constant => {
                const constant = ir.nodeData(node, Node.Constant);

                try writer.print("$0x{x:0>2} := op_constant: type: $0{x}, value: {}\n", .{ node.offset, constant.type.offset, constant.value_bits });
            },
            else => {},
        }
    }

    _ = try writer.print("\nInstructions:\n\n", .{});

    for (ir.node_list.items) |node| {
        _ = try writer.splatByteAll(' ', 4);

        switch (ir.nodeTag(node).*) {
            .null => {},
            .type_int => {},
            .type_float => {},
            .variable => {},
            .constant => {
                const constant = ir.nodeData(node, Node.Constant);

                try writer.print("$0x{x:0>2} := op_constant: {}\n", .{ node.offset, constant.value_bits });
            },
            .fadd => {
                const instruction = ir.nodeData(node, Node.FAdd);

                try writer.print("$0x{x:0>2} := op_fadd: ", .{node.offset});

                try ir.printOperand(writer, instruction.lhs);

                try writer.print(", ", .{});

                try ir.printOperand(writer, instruction.rhs);

                try writer.writeAll("\n");
            },
            .fsub => {
                const instruction = ir.nodeData(node, Node.FSub);

                try writer.print("$0x{x:0>2} := op_fsub: ", .{node.offset});

                try ir.printOperand(writer, instruction.lhs);

                try writer.print(", ", .{});

                try ir.printOperand(writer, instruction.rhs);

                try writer.writeAll("\n");
            },
            .fmul => {
                const fmul = ir.nodeData(node, Node.FAdd);

                try writer.print("$0x{x:0>2} := op_fmul: ", .{node.offset});

                try ir.printOperand(writer, fmul.lhs);

                try writer.print(", ", .{});

                try ir.printOperand(writer, fmul.rhs);

                try writer.writeAll("\n");
            },
            .fnegate => {
                const fnegate = ir.nodeData(node, Node.FNegate);

                try writer.print("$0x{x:0>2} := op_fnegate: ", .{node.offset});

                try ir.printOperand(writer, fnegate.operand);

                try writer.writeAll("\n");
            },
            .return_value => {
                const return_value = ir.nodeData(node, Node.ReturnValue);

                try writer.print("$0x{x:0>2} := op_return: ", .{node.offset});

                try ir.printOperand(writer, return_value.return_value);

                try writer.writeAll("\n");
            },
        }
    }
}

pub fn printOperand(ir: Ir, writer: *std.Io.Writer, node: Node) !void {
    switch (ir.nodeTag(node).*) {
        .constant => {
            const constant = ir.nodeData(node, Node.Constant);

            switch (ir.nodeTag(constant.type).*) {
                .type_int => {
                    const integer: u32 = constant.value_bits;

                    try writer.print("{}", .{integer});
                },
                .type_float => {
                    const float: f32 = @bitCast(constant.value_bits);

                    try writer.print("{}", .{float});
                },
                else => {},
            }
        },
        else => {
            try writer.print("$0x{x}", .{node.offset});
        },
    }
}

const std = @import("std");
const Ir = @This();
