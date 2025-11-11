//! A sea of nodes representation for spirv semantics

node_buffer: std.ArrayList(u32) = .{},
effect_nodes: std.ArrayList(Node) = .{},
node_list: std.ArrayList(Node) = .{},
types: std.AutoArrayHashMapUnmanaged(Node, void) = .{},
constants: std.AutoArrayHashMapUnmanaged(Node.Constant, Node) = .{},

node_dedup_map: std.StringArrayHashMapUnmanaged(Node) = .{},

pub fn buildNodeOpConstant(
    ir: *Ir,
    allocator: std.mem.Allocator,
    constant: Node.Constant,
) !Node {
    return try ir.appendNode(allocator, .constant, Node.Constant, constant);
}

pub fn buildNodeOpFAdd(
    ir: *Ir,
    allocator: std.mem.Allocator,
    result_type: Node,
    lhs: Node,
    rhs: Node,
) !Node {
    if (ir.nodeTag(lhs).* == .constant and ir.nodeTag(rhs).* == .constant) {
        const lhs_value = ir.nodeData(lhs, Node.Constant);
        const rhs_value = ir.nodeData(rhs, Node.Constant);

        const addition = lhs_value.value(f32) + rhs_value.value(f32);

        return try ir.appendNode(allocator, .constant, Node.Constant, .{
            .type = lhs_value.type,
            .value_bits = @bitCast(addition),
        });
    }

    const result_node = try ir.appendNode(allocator, .fadd, Node.FAdd, .{
        .type = result_type,
        .lhs = lhs,
        .rhs = rhs,
    });

    return result_node;
}

pub fn buildNodeOpFSub(
    ir: *Ir,
    allocator: std.mem.Allocator,
    result_type: Node,
    lhs: Node,
    rhs: Node,
) !Node {
    if (ir.nodeTag(lhs).* == .constant and ir.nodeTag(rhs).* == .constant) {
        const lhs_value = ir.nodeData(lhs, Node.Constant);
        const rhs_value = ir.nodeData(rhs, Node.Constant);

        const addition = lhs_value.value(f32) - rhs_value.value(f32);

        return try ir.appendNode(allocator, .constant, Node.Constant, .{
            .type = lhs_value.type,
            .value_bits = @bitCast(addition),
        });
    }

    const result_node = try ir.appendNode(allocator, .fsub, Node.FSub, .{
        .type = result_type,
        .lhs = lhs,
        .rhs = rhs,
    });

    return result_node;
}

pub fn buildNodeOpFMul(
    ir: *Ir,
    allocator: std.mem.Allocator,
    result_type: Node,
    lhs: Node,
    rhs: Node,
) !Node {
    if (ir.nodeTag(lhs).* == .constant and ir.nodeTag(rhs).* == .constant) {
        const lhs_value = ir.nodeData(lhs, Node.Constant);
        const rhs_value = ir.nodeData(rhs, Node.Constant);

        const addition = lhs_value.value(f32) * rhs_value.value(f32);

        return try ir.appendNode(allocator, .constant, Node.Constant, .{
            .type = lhs_value.type,
            .value_bits = @bitCast(addition),
        });
    }

    const result_node = try ir.appendNode(allocator, .fmul, Node.FMul, .{
        .type = result_type,
        .lhs = lhs,
        .rhs = rhs,
    });

    return result_node;
}

pub fn buildNodeOpFNegate(
    ir: *Ir,
    allocator: std.mem.Allocator,
    result_type: Node,
    operand: Node,
) !Node {
    if (ir.nodeTag(operand).* == .constant) {
        const operand_value = ir.nodeData(operand, Node.Constant);

        const negation = -operand_value.value(f32);

        return try ir.appendNode(allocator, .constant, Node.Constant, .{
            .type = operand_value.type,
            .value_bits = @bitCast(negation),
        });
    }

    const result_node = try ir.appendNode(allocator, .fnegate, Node.FNegate, .{
        .type = result_type,
        .operand = operand,
    });

    return result_node;
}

pub fn appendNode(
    ir: *Ir,
    allocator: std.mem.Allocator,
    node_tag: Node.Tag,
    T: type,
    node_data: T,
) !Node {
    const instruction = &node_data;

    const node_bytes = std.mem.asBytes(instruction);

    const node_query = try ir.node_dedup_map.getOrPut(allocator, node_bytes);

    if (node_query.found_existing) {
        return node_query.value_ptr.*;
    }

    const words_to_append = std.math.divCeil(comptime_int, @sizeOf(T), @sizeOf(u32)) catch @compileError("Type T have size divisible by 4 bytes");

    const node_offset: usize = ir.node_buffer.items.len;
    try ir.node_buffer.appendNTimes(allocator, undefined, words_to_append);

    const node: Node = .{ .offset = @intCast(node_offset) };

    ir.nodeData(node, T).* = node_data;

    switch (node_tag) {
        .return_value => {
            try ir.effect_nodes.append(allocator, node);
        },
        else => {},
    }

    node_query.value_ptr.* = node;

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
        null,

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

    pub const Data = union(Tag) {
        null: void,
        type_int: TypeInt,
        type_float: TypeFloat,
        constant: Constant,
        variable: Variable,
        fadd: FAdd,
        fsub: FSub,
        fmul: FMul,
        fnegate: FNegate,
        return_value: ReturnValue,
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

        pub fn value(constant: Constant, comptime T: type) T {
            return @bitCast(constant.value_bits);
        }
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
